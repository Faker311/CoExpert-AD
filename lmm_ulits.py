import os
from PIL import Image
import numpy as np
import json
import jsonlines
import torch
import matplotlib.pyplot as plt
import glob
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
import math

'''============= File Processing Helpers ============='''
def make_convs(convs,img_num=0):
    '''
    make_convs -> dict convert a list of convs to dict to meet limit of llava train data
    eg:
    [ q1, a1, q2, a2, ...] -> [{"from": "human", "value": q1}, {"from": "gpt", "value": a1}, ...]
    '''
    data_convs=[]
    roles=["human","gpt"]
    for i,c in enumerate(convs):
        if i==0:
            if "<image>" not in c:
                c="".join(["<image>\n"]*img_num)+c
        data_convs.append({"from":roles[i%2],"value":c})
    return data_convs

def make_train_data_json(data_id,img_ps,convs,img_p=None,masks=None,bboxes=None,**kwargs):
    '''
    make_train_data_json -> dict produce dict in format of llava train data
    data_id -> str : corresponding "id" in llava train data
    img_ps -> list[str] : list of image paths will be store in "src", note if only 1 image, "image" will replace "src"
    convs -> list[str] : list of conversations in format of [ q1, a1, q2, a2, ...], will be auto converted to llava train data format
    img_p -> str : image path that saved in "image", will use "src"[0] if set to None
    masks -> list[str] : mask paths that saved in "masks"
    bboxes -> list[list[list[int]]] : bboxes saved in "bboxes" for every image, every bbox, every axis
    **kwargs -> Any : any other kwargs will auto convert to key-value in data dict
    '''
    data={}
    data["id"]=data_id
    if type(img_ps) is str:
        img_ps=[img_ps]
    if len(img_ps)>1:
        data["src"]=img_ps
    if img_p is None:
        data["image"]=img_ps[0]
    else:
        data["image"]=img_p
    data["image_num"]=img_num=len(img_ps)
    data["conversations"]=make_convs(convs,img_num)
    if masks is not None:
        if type(masks) is list and len(masks)==1:
            masks=masks[0]
        data["mask"]=masks
    if bboxes is not None:
        data["bbox"]=bboxes
    data.update(kwargs)
    return data

def data_read(data_path):
    '''
    data_read -> list[dict] used to read data from question.jsonl or other json file storing eval data annotations, 
    current support llava train data format and question.jsonl format, after read data should be in llava train data format
    data_path -> str : the full path of the eval data file you want to read
    '''
    data_dir=os.path.dirname(data_path)
    dataset_name=os.path.basename(data_dir)
    all_datas=[]
    if data_path.endswith(".jsonl"): 
        with jsonlines.open(data_path) as f:
            jsonl_datas=list(f)
        for i in jsonl_datas:
            question_id=i["question_id"]
            img_p=os.path.join("imgs",i["image"])
            text=i["text"]
            ref=i.get("reference",None)
            if "sub_dataset" in i:
                subset=i["sub_dataset"] # in old formation it is sub_dataset
            else: # try to guess a subset
                default_subset=i["origin_path"].split("/")[0]
                if default_subset==dataset_name:
                    default_subset=i["origin_path"].split("/")[1]
                subset=i.get("subset", default_subset)
            gt=i.get("gt",None)
            origin_path=i["origin_path"]
            mask_p=i.get("mask",None)
            if mask_p is not None:
                mask_p=os.path.join(mask_p)
            bbox=i.get("bbox",None)
            if ref is not None:
                ref=os.path.join("mvtec",ref)
                img_ps=[img_p,ref]
            else:
                img_ps=[img_p]
            all_datas.append(make_train_data_json(question_id,img_ps,[text,""],img_ps[0],mask_p,bbox,subset=subset,gt=gt,origin_path=origin_path))
    else: # llava train data formation, still need "subset" "gt" "origin_path" just for unified mid format
        with open(data_path) as f:
            all_datas=json.load(f)
        for i in all_datas:
            if "subset" not in i:
                i["subset"]=i["image"].split("/")[1]
            if "gt" not in i:
                i["gt"]=0 if "/good/" in i["image"] else 1
            if "origin_path" not in i:
                i["origin_path"]=i["image"]
            if "bbox" not in i:
                i["bbox"]=None
            if "mask" not in i:
                i["mask"]=None
    return all_datas
    
def data_write(ans,out_path,format="json",mode="w"):
    '''
    data_write -> None write answers to json or jsonl file for mid storage, generally jsonl for mid temp storage and json for final result.
    ans -> list[dict] : all answers you want to save
    out_path -> str : full path of answer file path
    format -> ["json","jsonl"] : the out file format
    mode -> ["a","w"] : write mode of file
    '''
    ans_dir=os.path.dirname(out_path)
    os.makedirs(ans_dir,exist_ok=True)
    if format=="jsonl":
        out_path=out_path.replace(".json",".jsonl")
        with jsonlines.open(out_path,mode) as f:
            f.write(ans)
    else:
        out_path=out_path.replace(".jsonl",".json")
        with open(out_path,mode) as f:
            json.dump(ans,f,indent=4)

def load_and_resize(input_source, target_width=512, mode='RGB', return_numpy=False):
    """
    Generic loading and resizing helper.
    Args:
        input_source (str or np.ndarray): file path or numpy array.
        target_width (int): target width.
        mode (str): 'RGB' (for source image) or 'L' (for mask).
        return_numpy (bool): whether to return a numpy array.
        
    Returns:
        PIL.Image or np.ndarray
    """
    img = None
    # --- 1. Load input ---
    if isinstance(input_source, str):
        # A. Handle NumPy files (.npy, .npz)
        if input_source.endswith(('.npy', '.npz')):
            data = np.load(input_source)
            if isinstance(data, dict):
                data = data["anomaly_map"] # Keep original behavior.
            # Convert NumPy to PIL for unified resize processing.
            # Float maps and integer masks are handled by Image.fromarray + mode.
            if mode == 'L':
                img = Image.fromarray(data.astype('uint8'), mode=mode)
            else:
                img = Image.fromarray(data)
                
        # B. Handle image files (.png, .jpg, etc.)
        else:
            img = Image.open(input_source).convert(mode)
            
    elif isinstance(input_source, np.ndarray):
        # C. Direct NumPy array input.
        img = Image.fromarray(input_source.astype('uint8'), mode=mode)
        
    else:
        raise ValueError(f"Unsupported input type: {type(input_source)}")

    # --- 2. Resize while preserving aspect ratio ---
    w, h = img.size
    if w != target_width:
        ratio = target_width / w
        target_height = int(h * ratio)
        
        # Choose interpolation method by mode.
        # 'RGB' -> LANCZOS for smooth high-quality resizing.
        # 'L' -> NEAREST to avoid introducing new mask label values.
        resample_method = Image.Resampling.NEAREST if mode == 'L' else Image.Resampling.LANCZOS
        
        img = img.resize((target_width, target_height), resample_method)

    # --- 3. Return output ---
    if return_numpy:
        return np.array(img)
    else:
        return img

def load_mmad_data(json_path, one_shot:bool=False, ref_mode:str="similar", ref_num:int=0):
    '''
    Read mmad.json and convert it to a standardized list of dictionaries.
    Each dictionary contains keys like: ["subset", "image", "bbox", "conversations", "origin_path", "ref_image", ...]
    
    Args:
        json_path (str): JSON file path.
        one_shot (bool): whether one-shot mode is enabled. Default: False.
        mode (str): reference-image loading mode, "random" or "similar". Default: "similar".
        topk (int): number of reference images to load. Default: 1.
    '''
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_datas = []
    # 2. Iterate over each sample.
    # key: global path ID (e.g., "DS-MVTec/bottle/image/broken_large/000.png")
    # value: dictionary containing detailed sample metadata.
    for unique_id, val in data.items():
        content = {}
        parts = unique_id.split("/")
        content["dataset"] = parts[0] if len(parts) > 0 else "unknown" # Set ["dataset"].
        content["subset"] = parts[1] if len(parts) > 1 else "unknown" # Set ["subset"].
        content["bbox"] = val.get("bbox", None) # Set ["bbox"].
        
        # Load image, mask, and semantic-segmentation mask paths.
        raw_image = val.get("image_path", "")
        raw_mask = val.get("mask_path", None)
        # Rewrite paths for MVTec-LOCO to match local data layout.
        if unique_id.startswith("MVTec-LOCO") or unique_id.startswith("FA-AD"):
            content["image"] = os.path.join("MVTec-LOCO-AD-512/orig_512", content["subset"], raw_image).replace("\\", "/")
            if raw_mask is not None:
                content["mask"] = os.path.join("MVTec-LOCO-AD-512/orig_512_mask", content["subset"], raw_image).replace("\\", "/")
            else:
                # content["mask"] = raw_mask # For normal test samples, no defect mask is required.
                content["mask"] = os.path.join("MVTec-LOCO-AD-512/orig_512_mask", content["subset"], raw_image).replace("\\", "/") # Use class segmentation mask for normal test samples.
            content["seg_masks"] = os.path.join("data/MVTec-LOCO-AD-512/orig_512_seg", content["subset"], raw_image).replace("\\", "/")
            config = DATASET_CONFIG[content["subset"]]
            content["size_bd"] = config["box"] # Store segmentation boundary for MVTec-LOCO rectangle drawing.
        # elif unique_id.startswith("MVTec-AD"):
        else:
            content["image"] = unique_id
            if raw_mask is not None:
                if "musc" in raw_mask.lower():
                    content["mask"] = raw_mask
                else:
                    content["mask"] = os.path.join(content["dataset"], content["subset"], raw_mask).replace("\\", "/")
            else:   
                content["mask"] = raw_mask   
        
        # --- Added logic: set content["ref_image"] ---
        content["ref_image"] = None # Default initialization.
        if one_shot:
            # 1. Select template list by ref_mode.
            template_key = "similar_templates" if ref_mode == "similar" else "random_templates"
            ref_list = val.get(template_key, []) 
            # 2. Keep top ref_num references.
            if ref_list and isinstance(ref_list, list):
                selected_refs = ref_list[:ref_num]
                # 3. Format reference image paths.
                formatted_refs = []
                for ref_raw_path in selected_refs:
                    # For MVTec-LOCO, references are often filenames and require path rewriting.
                    if unique_id.startswith("MVTec-LOCO"):
                        new_ref = ref_raw_path.replace("MVTec-LOCO", "MVTec-LOCO-AD-512/orig_512", 1)
                        formatted_refs.append(new_ref)
                    else:
                        formatted_refs = selected_refs
                        break
                    
                if formatted_refs:
                    content["ref_image"] = formatted_refs
        # --- End added logic ---
        
        raw_conversations = val.get("conversation", [])
        clean_conversations = []
        answers_list = []
        types_list = []
        conv_texts = []  # will hold [q1, a1, q2, a2, ...]
        gt_value = 0

        for item in raw_conversations:
            item_copy = item.copy()
            ans_key = item_copy.pop("Answer", "")
            clean_conversations.append(item_copy)
            answers_list.append(ans_key)
            types_list.append(item_copy.get("type", None))   # Save each question type to types_list.

            question = item_copy.get("Question", "")
            options = item_copy.get("Options", {})
            if options:
                opts_lines = [f"{k}: {v}" for k, v in options.items()]
                question_with_options = "Question:\n" + question + ("\nOptions:\n" + "\n".join(opts_lines) if question else "\n".join(opts_lines))
            else:
                question_with_options = question
                
            ans_text = options.get(ans_key, "")

            conv_texts.append(question_with_options)
            conv_texts.append(ans_key)

            if item_copy.get("type") == "Anomaly Detection" and "Yes" in ans_text:
                gt_value = 1

        # build conversations in the same format as data_read (use make_convs)
        content["conversations"] = make_convs(conv_texts, img_num=0)
        content["answers"] = answers_list
        content["types"] = types_list
        
        # 4. Other metadata fields.
        content["question_id"] = unique_id
        content["origin_path"] = unique_id
        content["ad_class"] = "logical" if (("mvtec-loco" in content["origin_path"].lower()) or "fa-ad" in content["origin_path"].lower()) and ("structural" not in content["origin_path"].lower()) else "structural" # Split ad_class by whether origin_path contains "structural".

        content["gt"] = gt_value
        
        all_datas.append(content)
        
    return all_datas

'''============= 以下为类别分割掩码处理相关函数 ============='''
def show_mask_classes(img_path, box=None):
    """
    展示指定矩形框内的原始掩码与按类别分割的二值掩码（黑底 + 类别色填充）。
    参数:
      img_path: 掩码图路径（P 模式或 RGB）
      box: None 或 [x1,y1,x2,y2]，像素坐标，左上包含，右下不包含。若为 None 则为整图。
    输出:
      使用 matplotlib 弹出窗口显示裁剪区域原图与各类别掩码，子图标题含类别编号、像素数与占比（相对于框面积）。
    """
    # 1. Color palette.
    palette = [0, 0, 0, 204, 241, 227, 112, 142, 18, 254, 8, 23, 207, 149, 84, 202, 24, 214,
        230, 192, 37, 241, 80, 68, 74, 127, 0, 2, 81, 216, 24, 240, 129, 20, 215, 125, 161, 31, 204,
        254, 52, 116, 117, 198, 203, 4, 41, 68, 127, 252, 61, 21, 3, 142, 40, 10, 159, 241, 61, 36,
        14, 175, 77, 144, 61, 115, 131, 79, 97, 109, 177, 163, 58, 198, 140, 17, 235, 168, 47, 128, 91,
        238, 103, 45, 124, 35, 228, 101, 48, 232, 74, 124, 114, 78, 49, 30, 35, 167, 27, 137, 231, 47,
        235, 32, 39, 56, 112, 32, 62, 173, 79, 86, 44, 201, 77, 47, 217, 246, 223, 57]
    palette = palette + [0] * (768 - len(palette))
    palette_arr = np.array(palette).reshape(-1, 3)

    img = load_and_resize(img_path, target_width=512, mode="P")  # Resize to width 512 for consistency.
    # Parse index matrix arr (class indices).
    if img.mode == "P" or img.mode == "L":
        arr = np.array(img)
        safe_arr = np.clip(arr, 0, 255)
        original_vis_img = palette_arr[safe_arr].astype(np.uint8)
    else:
        rgb = np.array(img.convert("RGB"))
        original_vis_img = rgb
        h_rgb, w_rgb, _ = rgb.shape
        flat = rgb.reshape(-1, 3)
        _, inv = np.unique(flat, axis=0, return_inverse=True)
        arr = inv.reshape(h_rgb, w_rgb)

    h, w = arr.shape

    # Parse and clamp box.
    if box is None:
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            print("Warning: invalid box, using whole image.")
            x1, y1, x2, y2 = 0, 0, w, h

    crop_w, crop_h = x2 - x1, y2 - y1
    box_area = crop_w * crop_h
    if box_area == 0:
        print("Warning: box area is zero, nothing to display.")
        return

    # Crop original visualization and index matrix.
    arr_crop = arr[y1:y2, x1:x2]
    orig_crop_vis = original_vis_img[y1:y2, x1:x2]

    # Use classes from 0 to max_idx to keep deterministic order.
    max_idx = int(np.max(arr))
    classes = list(range(0, max_idx + 1))

    total_plots = len(classes) + 1  # +1 for cropped original view.
    cols = min(5, total_plots)
    rows = (total_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).reshape(-1) if total_plots > 1 else np.array([axes])

    # Subplot 0: cropped original mask visualization.
    ax0 = axes[0]
    ax0.imshow(orig_crop_vis)
    ax0.set_title(f"Crop {x1},{y1},{x2},{y2}  (w={crop_w},h={crop_h})")
    ax0.axis('off')

    # Draw class masks within the cropped area.
    for i, cls in enumerate(classes):
        ax = axes[i + 1]
        mask = (arr_crop == cls)
        px = int(mask.sum())
        prop = px / box_area
        # Black background with class color fill.
        vis_img = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        color = palette_arr[cls] if cls < len(palette_arr) else np.array([255, 255, 255], dtype=np.uint8)
        if px > 0:
            vis_img[mask] = color
        ax.imshow(vis_img)
        ax.set_title(f"class={int(cls)} px={px} ({prop:.2%})\nRGB={list(map(int,color))}")
        ax.axis('off')

    # Hide unused subplots.
    for j in range(total_plots, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"Mask Crop Analysis: {os.path.basename(img_path)}", fontsize=14)
    fig.tight_layout()
    plt.show()

def read_seg_mask(img_path, num_cls):
    """
    读取分割掩码并返回 shape=(num_cls, H, W) 的 one-hot 张量。
    - 支持 P/L（索引）和 RGB（根据颜色唯一值映射为索引）输入。
    - 若 num_cls >= 掩码实际类别数：正常返回全部类别。
    - 若 num_cls < 掩码实际类别数：把类别 idx >= num_cls 的像素视为 0（忽略），只返回前 num_cls 组掩码。
    """
    img = load_and_resize(img_path, target_width=512, mode='P')  # Resize to width 512 for consistency.
    # Get index matrix arr (H, W).
    if img.mode in ("P", "L"):
        arr = np.array(img)
    else:
        rgb = np.array(img.convert("RGB"))
        h_rgb, w_rgb, _ = rgb.shape
        flat = rgb.reshape(-1, 3)
        _, inv = np.unique(flat, axis=0, return_inverse=True)
        arr = inv.reshape(h_rgb, w_rgb)
    arr = arr.astype(np.int64)

    if num_cls < 1:
        raise ValueError("num_cls must be >= 1")

    max_idx = int(arr.max()) if arr.size > 0 else -1
    if max_idx >= num_cls:
        # Map out-of-range classes to 0 (ignored); keep only first num_cls classes.
        arr = arr.copy()
        arr[arr >= num_cls] = 0

    H, W = arr.shape
    idx = torch.from_numpy(arr).long().unsqueeze(0)  # shape (1, H, W)
    onehot = torch.zeros((num_cls, H, W), dtype=torch.float)
    onehot.scatter_(0, idx, 1.0)

    return onehot

def single_process_mask(onehot_img, box=None, min_pixels=1, mode_class="SCENE2"):
    """
    统计 One-Hot 图像中除类别0以外的信息。
    
    参数:
      onehot_img: torch.Tensor, shape (num_cls, H, W)
      box: [x1, y1, x2, y2]
      is_rel: bool
        - False: 返回绝对坐标 (x, y)，占比为 (该类像素/Box面积)
        - True : 返回极坐标 (rho, theta)，占比为 (该类像素/所有Mask总像素)
                 rho = 距离/图宽, theta = 弧度
    """
    # Count metric: connected components or global/relative proportion. Position metric: relative polar center, absolute center, or shape spread.

    num_cls, h, w = onehot_img.shape
    
    # --- 1. Determine statistics region ---
    if box is None:
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            # print("Warning: invalid box, treating as whole image.")
            x1, y1, x2, y2 = 0, 0, w, h

    box_area = (x2 - x1) * (y2 - y1)
    if box_area == 0:
        return []

    stats_list = []
    
    # --- 2. Collect raw data ---
    raw_data = {}
    total_mask_pixels = 0 # Total detected class pixels (used as denominator for relative proportion).
    for cls_idx in range(1, num_cls):
        mask = onehot_img[cls_idx]
        mask_crop = mask[y1:y2, x1:x2]
        pixel_count = int(mask_crop.sum().item())
        if pixel_count == 0 or pixel_count < min_pixels:
            continue
        total_mask_pixels += pixel_count
        indices = torch.nonzero(mask_crop)
        if indices.shape[0] > 0:
            y_mean = indices[:, 0].float().mean().item() + y1
            x_mean = indices[:, 1].float().mean().item() + x1
            center = (x_mean, y_mean)
        else:
            center = (float(x1), float(y1))
        raw_data[cls_idx] = {
            'center': center,
            'count': pixel_count,
            'name': f"obj_{cls_idx}"
        }

    # --- 3. Generate results ---
    # Sort output by class index.
    sorted_keys = sorted(raw_data.keys())

    if mode_class == "SCENE1":
        # === B. Foreground relative proportion && A. Relative polar centroid ===
        # Get baseline object (obj_1).
        ref_exists = 1 in raw_data
        if ref_exists: 
            ref_center = raw_data[1]['center']
            ref_mask_pixels = raw_data[1]['count']
        else: 
            ref_center = None
            ref_mask_pixels = 0

        for cls_idx in sorted_keys:
            data = raw_data[cls_idx]
            cur_center = data['center']
            
            # A. Relative polar centroid.
            if ref_exists:
                # dx, dy: displacement relative to obj_1.
                dx = cur_center[0] - ref_center[0]
                dy = cur_center[1] - ref_center[1]
                # rho: normalized distance (distance / image width).
                dist_px = math.hypot(dx, dy)
                rho = dist_px / w if w > 0 else 0
                # theta: angle in radians (-pi to pi); for obj_1 itself dx=0, dy=0 -> theta=0.0.
                theta = math.atan2(dy, dx)
                
                pos_tuple = (round(rho, 3), round(theta, 3))
            else:
                pos_tuple = (0.0, 0.0) # If baseline is missing, default to zeros.

            # B. Relative mask proportion (except obj_1).
            # rel_proportion = data['count'] / total_mask_pixels if total_mask_pixels > 0 else 0
            rel_proportion = data['count'] / ref_mask_pixels if ref_mask_pixels > 1000 else 0
            prop_val = round(rel_proportion, 3)

            stats_list.append((data['name'], pos_tuple, prop_val))

    elif mode_class == "SCENE2":
        # === B. Connected-component count + A. Shape spread metrics ===
        for cls_idx in sorted_keys:
            data = raw_data[cls_idx]
            # Recompute mask here for image-moment calculation to reduce memory usage.
            # 1. Extract mask and convert to NumPy for cv2 operations.
            mask_tensor = onehot_img[cls_idx][y1:y2, x1:x2]
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8) # Ensure uint8 type (0/255 or 0/1).
            # 2. Compute image moments.
            M = cv2.moments(mask_np, binaryImage=True)
            if M["m00"] > 0:
                # Normalized central moments (remove translation/scale effects, keep shape distribution).
                mu20 = M["mu20"] / M["m00"]
                mu02 = M["mu02"] / M["m00"]
                mu11 = M["mu11"] / M["m00"]
                # Build covariance terms and compute eigenvalues lambda1, lambda2.
                # cov = [[mu20, mu11], [mu11, mu02]]
                # Use closed-form eigenvalue computation.
                delta = math.sqrt(4 * mu11**2 + (mu20 - mu02)**2)
                l1 = (mu20 + mu02 + delta) / 2
                l2 = (mu20 + mu02 - delta) / 2
                
                # A. Shape-ratio dispersion.
                # Shape ratio: sqrt(l2/l1). 0=line-like, 1=circle/square-like.
                shape_ratio = math.sqrt(l2 / l1) if l1 > 1e-6 else 0
                # Spread: l1 + l2, reflecting the spatial spread of distribution.
                spread = l1 + l2
            else:
                shape_ratio = 0.0
                spread = 0.0

            metric_tuple = (round(shape_ratio, 3), int(spread))

            # B. Connected-component count in mask.
            # connectivity=8 means 8-neighborhood connectivity (diagonals included).
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
            obj_count = 0
            min_blob_area = 1000  # Minimum area threshold to filter tiny noisy blobs.
            for i in range(1, num_labels): # num_labels includes background (0), so candidates are 1..num_labels-1.
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_blob_area:
                    obj_count += 1
            count_int = obj_count

            stats_list.append((data['name'], metric_tuple, count_int))
    else:
        # B. Global absolute proportion + A. Absolute pixel centroid ===
        for cls_idx in sorted_keys:
            data = raw_data[cls_idx]
            
            # A. Absolute pixel centroid.
            pos_tuple = (int(data['center'][0]), int(data['center'][1]))
            
            # B. Global absolute proportion (relative to full box area).
            abs_proportion = data['count'] / box_area
            prop_val = round(abs_proportion, 3)

            stats_list.append((data['name'], pos_tuple, prop_val))

    # print(f"统计结果 (catalogy={mode_class}): {stats_list}")
    return stats_list

def batch_process_masks(folder_path, num_cls, box=None, min_pixels=1):
    """
    遍历文件夹下的所有图像文件，执行 read_mask 和 single_process_mask
    """
    # 1. 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹不存在 -> {folder_path}")
        return

    # 2. 获取所有图片文件 (支持 png, jpg, jpeg, bmp, tif)
    # 使用 set 去重，避免大小写重复
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
    image_files = []
    
    for ext in extensions:
        # 递归搜索或仅搜索当前目录，这里假设仅搜索当前目录
        # os.path.join(folder_path, ext) 会构建如 ".../*.png"
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        print(f"在该文件夹下未找到图像文件: {folder_path}")
        return

    print(f"共找到 {len(image_files)} 张图像，开始处理...\n")
    print("="*50)

    results_summary = {}

    # 3. 遍历处理
    for idx, img_path in enumerate(image_files):
        file_name = os.path.basename(img_path)
        print(f"[{idx+1}/{len(image_files)}] 处理文件: {file_name}")
        mode_class = img_path.replace('\\', '/').split('/')[-4]  # 假设类别名在倒数第二级目录
        
        try:
            # --- 调用之前的 read_seg_mask ---
            # 注意：请确保 mask 里的像素值 < num_cls，否则 scatter 会报错
            onehot_tensor = read_seg_mask(img_path, num_cls)
            
            # --- 调用之前的 single_process_mask ---
            # 这会打印该图片的统计信息
            stats = single_process_mask(onehot_tensor, box=box, min_pixels=min_pixels, mode_class=mode_class)
            
            # 收集结果
            results_summary[file_name] = stats
            
        except Exception as e:
            print(f"!!! 处理 {file_name} 时发生错误: {e}")
        
        print("-" * 30) # 分隔线

    return results_summary

def data_statistics(results_summary, output_dir="output_stats"):
    """
    聚合统计结果并进行可视化绘制
    修改点：右侧图表改为散点图展示占比分布
    """
    if not results_summary:
        print("没有数据可供统计。")
        return

    # 1. 数据聚合
    # stats_data[obj_name] = {'x': [], 'y': [], 'prop': []}
    stats_data = defaultdict(lambda: {'x': [], 'y': [], 'prop': []})
    min_x, min_y, max_x, max_y = 0, 0, 0, 0

    print(f"正在聚合 {len(results_summary)} 个文件的统计数据...")

    for filename, items in results_summary.items():
        for item in items:
            # item: ('obj_1', (150, 200), 0.125)
            obj_name, pos, prop_val = item
            x, y = pos
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

            stats_data[obj_name]['x'].append(x)
            stats_data[obj_name]['y'].append(y)
            stats_data[obj_name]['prop'].append(prop_val)

    if not stats_data:
        print("未检测到任何类别的目标信息。")
        return

    # 2. 创建可视化图表
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.colormaps.get_cmap('tab20') 

    # --- 左图: 空间重心分布 (保持不变) ---
    ax_spatial = axes[0]
    for idx, (obj_name, data) in enumerate(stats_data.items()):
        ax_spatial.scatter(data['x'], data['y'], 
                           label=f"{obj_name}", 
                           color=colors(idx), alpha=0.7, s=50, edgecolors='None')
    
    ax_spatial.set_title("Spatial Distribution")
    ax_spatial.set_xlabel("X (Value)")
    ax_spatial.set_ylabel("Y (Value)")
    ax_spatial.invert_yaxis() # 坐标原点在左上角
    ax_spatial.legend(loc='upper right')
    # 动态调整坐标范围
    ax_spatial.set_xlim(min_x, max_x)
    ax_spatial.set_ylim(min_y, max_y)

    # --- 右图: 像素占比散点图 (修改部分) ---
    ax_prop = axes[1]
    
    obj_names = list(stats_data.keys())
    
    # 遍历每个类别进行绘制
    for idx, obj_name in enumerate(obj_names):
        props = stats_data[obj_name]['prop']
        
        # X轴坐标生成：
        # 使用 idx 作为基准 x 坐标 (0, 1, 2...)
        # 加上 np.random.normal 添加随机抖动，防止点完全重叠
        # scale=0.05 控制抖动宽度
        x_jitter = np.random.normal(loc=idx, scale=0.05, size=len(props))
        
        ax_prop.scatter(x_jitter, props, 
                        label=f"{obj_name} (n={len(props)})",
                        color=colors(idx), 
                        alpha=0.6,  # 设置透明度，重叠处颜色会变深
                        s=60)       # 点的大小

    # 设置 X 轴标签为类别名称
    ax_prop.set_xticks(range(len(obj_names)))
    ax_prop.set_xticklabels(obj_names)
    
    # 设置网格和标题
    ax_prop.set_title("Defect Pixel Proportion (Scatter)")
    ax_prop.set_ylabel("Proportion (%)")
    ax_prop.set_xlabel("Defect Class")
    ax_prop.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 这里的 Legend 主要用于显示样本数量 n
    ax_prop.legend(loc='upper right')

    # 3. 保存与显示
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "statistics_scatter_result.png")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"统计图表已保存至: {os.path.abspath(save_path)}")
    plt.show()

def save_reference_model(summary, save_path):
    """
    将统计结果保存为 JSON 文件
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            # indent=4 让文件格式化显示，方便人工阅读
            json.dump(summary, f, ensure_ascii=False, indent=4)
        print(f"基准模型已保存至: {save_path}")
    except Exception as e:
        print(f"保存失败: {e}")

def load_reference_model(load_path):
    """
    从 JSON 文件加载统计结果
    """
    if not os.path.exists(load_path):
        print(f"文件不存在: {load_path}")
        return None
    
    try:
        with open(load_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        return summary
    except Exception as e:
        print(f"加载失败: {e}")
        return None

def get_context_from_path(img_path):
    """
    解析文件路径，返回对应的配置字典、类别名称和构建的良品训练集路径。
    假设目录结构遵循 MVTec 格式: .../category_name/test/type/xxx.png
    """
    normalized_path = img_path.replace('\\', '/')
    
    # 1. 识别类别名称
    category = None
    for key in DATASET_CONFIG.keys():
        if f"/{key}/" in normalized_path or key in normalized_path.split('/'):
            category = key
            break
    
    if category is None:
        print(f"[Warning] 路径中未识别到已知类别 (breakfast_box, pushpins, etc.): {img_path}")
        # 返回默认配置，防止报错
        return None

    config = DATASET_CONFIG[category]
    config['category_name'] = category
    
    return config

def analyze(current_stats, reference_summary, config, strictness=3.0):
    """
    参数:
      current_stats: single_process_mask 输出列表
      reference_summary: 历史汇总字典
      config: 从 DATASET_CONFIG 获取的当前类别配置字典
      strictness: 严谨度系数
    """
    target_ids = config.get("target_ids", [])
    label_map = config.get("label_map", {})
    check_rules = config.get("check_rules", {}) # 每个ID的检查策略 可选值: 'strict'(全查), 'pos_only'(只查位置), 'prop_only'(只查占比)
    
    # --- A. 建立基准分布 (同前，略微精简) ---
    ref_data = defaultdict(lambda: {'x': [], 'y': [], 'prop': []})
    for _, items in reference_summary.items():
        for item in items:
            obj_name, pos, prop_val = item
            # 过滤：如果历史数据里有不在 target_ids 的物体，可以选择忽略，也可以保留用于计算
            # 这里建议保留，以便建立完整的 mask 索引
            ref_data[obj_name]['x'].append(pos[0])
            ref_data[obj_name]['y'].append(pos[1])
            try: ref_data[obj_name]['prop'].append(prop_val)
            except: pass

    # 计算规则
    distribution_rules = {}
    for obj_name, data in ref_data.items():
        try:
            oid = int(obj_name.split('_')[1]) # 解析 obj_id, 如 "obj_1" -> 1
        except:
            continue
        if target_ids and oid not in target_ids: # 如果不在检测目标列表中，跳过规则生成（或者生成了也不检测）
            continue

        x_arr, y_arr, p_arr = np.array(data['x']), np.array(data['y']), np.array(data['prop'])
        if len(p_arr) < 2:
            distribution_rules[obj_name] = {
                'x_range': (min(x_arr), max(x_arr)),
                'y_range': (min(y_arr), max(y_arr)),
                'p_range': (min(p_arr), max(p_arr))
            }
        else:
            def get_bounds(arr, cfd=strictness):
                mean, std = np.mean(arr), np.std(arr)
                return (min(np.min(arr), mean - cfd*std), max(np.max(arr), mean + cfd*std))
            if config['category_name'] == "SCENE1" and obj_name == "obj_4":
                distribution_rules[obj_name] = {
                    'x_range': get_bounds(x_arr, 0.0),
                    'y_range': get_bounds(y_arr, 0.0),
                    'p_range': get_bounds(p_arr, strictness)
                    }
            else:
                distribution_rules[obj_name] = {
                    'x_range': get_bounds(x_arr, strictness),
                    'y_range': get_bounds(y_arr, strictness),
                    'p_range': get_bounds(p_arr, strictness)
                    }
    # print(f"Distribution Rules: {distribution_rules}")

    # --- B. 对比分析 ---
    curr_dict = {} # 将当前统计转为字典
    for item in current_stats:
        curr_dict[item[0]] = {'pos': item[1], 'prop': item[2]} # item: ('obj_1', (100,100), '0.12')
        
    analysis_results = [] # <--- 输出容器

    # 1. 遍历配置中要求的 Target IDs 进行检查
    for tid in target_ids:
        obj_key = f"obj_{tid}"
        display_name = label_map.get(tid, obj_key) # 获取中文名或默认名
        mode = check_rules.get(tid, 'strict') # 获取当前对象的检查模式，默认为 'strict'
        
        item_result = {
            "id": tid,
            "name": display_name,
            "status": "UNKNOWN", # 枚举: OK, MISSING, NG_QUALITY
            "issues": []
        }
        
        # 检查1: 是否存在 (存在性检查通常是必须的，除非策略允许缺失)
        if obj_key in distribution_rules and obj_key not in curr_dict:
            item_result["status"] = "MISSING"
            analysis_results.append(item_result)
            continue
        if obj_key not in distribution_rules:
            # 如果规则里没有（良品里本来就没有这个组件），跳过
            continue
            
        # 检查2: 检查质量 (Range Check)
        rule = distribution_rules[obj_key]
        curr = curr_dict[obj_key]
        item_result["prop"] = curr['prop']

        pos_error, prop_error = None, None
        # 2.1 独立判断位置是否异常
        if not (rule['x_range'][0] <= curr['pos'][0] <= rule['x_range'][1]) or \
           not (rule['y_range'][0] <= curr['pos'][1] <= rule['y_range'][1]):
            pos_error = f"location or distribution anomaly"
        
        # 2.2 独立判断占比是否异常
        if not (rule['p_range'][0] <= curr['prop'] <= rule['p_range'][1]):
            prop_error = f"quantity or proportion anomaly"
        
        # 2.3 根据配置策略 (mode) 决定是否记录错误
        if mode == 'strict': # 策略A: strict (严格模式，两样都必须达标)
            if pos_error: item_result["issues"].append(pos_error)
            if prop_error: item_result["issues"].append(prop_error)
        elif mode == 'pos_only': # 策略B: pos_only (只在乎位置，忽略占比错误)
            if pos_error: item_result["issues"].append(pos_error)
        elif mode == 'prop_only': # 策略C: prop_only (只在乎占比，忽略位置错误)
            if prop_error: item_result["issues"].append(prop_error)
        elif mode == 'flexible': # 策略D: flexible (宽松模式，只要有一个达标即可)
            if pos_error and prop_error: item_result["issues"].append(f"both position and proportion anomalies")
        
        # --- 输出结果 ---
        if item_result["issues"]:
            item_result["status"] = "NG_QUALITY" # 存在，但质量不好
        else:
            item_result["status"] = "OK" # 存在且符合规范

        analysis_results.append(item_result)
    return analysis_results

def describe(analysis_results, config):
    """
    参数:
      analysis_results: analyze 函数输出的列表
      config: 包含 valid_groups 的配置
    返回:
      summary_str: 格式化的总结字符串
    """
    # === 步骤 0: 数据准备 ===
    # 将列表转为字典，方便按 ID 索引
    # 结构: {1: {'status': 'OK', 'name': '螺丝', ...}, ...}
    res_map = {item['id']: item for item in analysis_results}
    
    valid_groups = config.get("valid_groups", [])
    # 如果未配置组合，默认将所有 target_ids 视为一个大组
    if not valid_groups:
        all_ids = config.get("target_ids", [])
        if all_ids:
            valid_groups = [all_ids]
        else:
            return "[FAILED] 配置无效 | 未定义检测目标"

    # === 步骤 1 & 2: 确认比较的组别 (竞选机制) ===
    # 核心逻辑：遍历所有可能的组，计算得分，得分最高的胜出
    # 评分标准 (Priority): 1. OK的数量最多  2. 存在的数量(不缺)最多
    
    best_group = None
    best_score = (-1, -1) # 初始分 (ok_count, present_count)
    
    for group in valid_groups:
        current_ok_count = 0
        current_present_count = 0
        c_ng_prop = 0.0 # NG 占比总和
        
        for tid in group:
            item = res_map.get(tid)
            if not item: continue # 防御性编程：防止配置ID不在结果里

            if item['status'] == 'OK':
                current_ok_count += 1
            
            elif item['status'] == 'NG_QUALITY':
                current_present_count += 1
                c_ng_prop += item.get('prop', 0.0)
                
        
        # 比较分数：元组比较机制会自动先比第一位，再比第二位
        # 例如: (3, 3) > (2, 5) -> OK多的优先
        # 例如: (2, 3) > (2, 2) -> OK一样多，存在多的优先
        current_score = (current_ok_count, current_present_count, c_ng_prop)
        
        if current_score > best_score:
            best_score = current_score
            best_group = group

    # 防御：如果 valid_groups 不为空但没选出任何组（极罕见），默认取第一组
    if best_group is None and valid_groups:
        best_group = valid_groups[0]

    # === 步骤 3: 总结描述 (仅针对 best_group) ===
    # 此时我们已经“认定”当前的图片属于 best_group 这一类产品
    # 只需要检查这一组里的零件是否都达标
    
    error_details = []
    
    for tid in best_group:
        item = res_map.get(tid)
        # 理论上 item 不应为空，除非 config 和 analyze 不对应
        if not item:
            error_details.append(f"class {tid} not found in analysis results")
            continue

        name = item['name']
        status = item['status']
        
        if status == 'MISSING':
            error_details.append(f"{name} | missing")
        elif status == 'NG_QUALITY':
            # 拼接具体的质量问题，如 "螺丝(位置偏离)"
            issues_str = ", ".join(item['issues'])
            error_details.append(f"{name} | {issues_str}")
        # 状态为 OK 的自动忽略，不记入错误列表

    # 生成最终字符串
    if not error_details:
        return f"The key logical characteristics fall within the reference ranges and there seem to be no obvious logical anomalies, but the details still require further verification."
    else:
        # 为每一项添加序号前缀
        numbered = [f"{i+1}. {s}" for i, s in enumerate(error_details)]
        err_str = ".\n  ".join(numbered)
        return f"Potential logical anomaly detected as follows:\n  {err_str}."

DATASET_CONFIG = {
# num_cls=class+1 ---> 
# breakfast_box=6+1 box=(0,0,512,410) 
# juice_bottle=9 box=(0,0,256,512) 
# pushpins=16+10 box=(0,0,512,300) 
# screw_bag=6+1 box=(0,0,512,352) 
# splicing_connectors=10 box=(0,0,512,256)

    "breakfast_box": {
        "min_pixels": 100,
        "num_cls": 7,          # 读取时容纳的最大类别数6+1 餐盘不考虑
        "box": (0, 0, 512, 410), # ROI [x1, y1, x2, y2]
        "target_ids": [1, 2, 3, 4, 5], # 只检测 obj_1 到 obj_5，忽略其他的辅助或背景类
        "label_map": { # ID 到名称的映射 (用于分析报告显示)
            1: "Orange", #橘子
            2: "Nectarine", #油桃
            3: "Granola", #燕麦
            4: "Banana slices", #香蕉片
            5: "Nuts" #坚果
            },
        # 每个检测目标的检查规则
        # strict默认值 prop_only 占比检查 pos_only 位置检查 loose宽松模式
        "check_rules": {i: f"strict" for i in range(1, 6)}
    },
    "juice_bottle": {
        "min_pixels": 2025,
        "num_cls": 9,
        "box": (0, 0, 256, 512),
        "target_ids": [1, 2, 3, 4, 6, 7, 8],
        "label_map": {
            1: "Orange label", # 橘子标签
            2: "Banana label", # 香蕉标签
            3: "Cherry label", # 樱桃标签
            4: "Bottom label", # 底部标签
            5: "Glass bottle mouth", # 玻璃瓶口
            6: "Orange juice", # 橘子果汁(橘色)
            7: "Banana juice", # 香蕉果汁(白色)
            8: "Cherry juice" # 樱桃果汁(红色)
        },
        "check_rules": {
            1: "strict",
            2: "strict",
            3: "strict",
            4: "strict",
            5: "prop_only",   # 瓶口位置检查
            6: "prop_only",  # 果汁占比检查
            7: "prop_only",
            8: "prop_only"
        },
        "valid_groups": [[1, 4, 6], [2, 4, 7], [3, 4, 8]] # 三种合法组合,
    },    
    "pushpins": {
        "min_pixels": 100,
        "num_cls": 16, # 15个图钉
        "box": (0, 0, 512, 300),
        "target_ids": list(range(1, 16)), 
        "label_map": {i: f"Pushpin_{i}" for i in range(1, 16)},
        "check_rules": {i: f"strict" for i in range(1, 16)}
    },
    "screw_bag": {
        "min_pixels": 100,
        "num_cls": 7, # 6+1 包装袋不考虑
        "box": (0, 0, 512, 352),
        "target_ids": [1, 2, 3, 4, 5],
        "label_map": {
            1: "Washers", # 垫圈
            2: "Nuts", # 螺母
            3: "Bolt heads", # 螺栓头
            4: "Shorter bolt", # 短螺栓
            5: "Longer bolt" # 长螺栓
            },
        "check_rules": {i: f"prop_only" for i in range(1, 6)}
    },
    "splicing_connectors": {
        "min_pixels": 400,
        "num_cls": 10,
        "box": (0, 0, 512, 256),
        "target_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "label_map": {
            1: "Left splicing connectors (2 pins)", # 2左侧接线端子
            2: "Yellow cable", # 黄色线缆
            3: "Right splicing connectors (2 pins)", # 2右侧接线端子
            4: "Left splicing connectors (5 pins)", # 5左侧接线端子
            5: "Red cable", # 红色线缆
            6: "Right splicing connectors (5 pins)", # 5右侧接线端子
            7: "Left splicing connectors (3 pins)", # 3左侧接线端子
            8: "Blue cable", # 蓝色线缆
            9: "Right splicing connectors (3 pins)", # 3右侧接线端子             
            },
        "check_rules": {i: f"strict" for i in range(1, 10)},
        "valid_groups": [[1, 2, 3], [4, 5, 6], [7, 8, 9]] # 三种合法组合,
    },
    "SCENE1": {
        "min_pixels": 50,
        "num_cls": 5,
        "box": (0, 0, 512, 512),
        "target_ids": [1, 2, 3, 4],
        "label_map": {
            1: "Bolt", # 螺栓
            2: "Nut", # 螺母
            3: "Gasket", # 垫片
            4: "Cotter pin", # 开口销            
            },
        "check_rules": {i: f"strict" for i in range(1, 5)},
    },
    "SCENE2": {
        "min_pixels": 500,
        "num_cls": 4,
        "box": (0, 0, 512, 512),
        "target_ids": [1, 2, 3],
        "label_map": {
            1: "Bolt heads with gaskets", # 螺栓头
            2: "Hex bolt head", # 六角螺栓
            3: "Hex nuts", # 六角螺母
            },
        "check_rules": {i: f"strict" for i in range(1, 4)},
    }
}
