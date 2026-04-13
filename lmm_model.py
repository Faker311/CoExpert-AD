import torch
import cv2
import numpy as np
import abc
import base64
import io
from openai import OpenAI
from lmm_ulits import *
import os  # Used for filesystem path operations.
import time
import sys
import json
from PIL import Image, ImageDraw, ImageFont
from peft import PeftModel

'''================================================= Image Utilities ================================================='''
def calc_IoU(a,b):
    calc_area=lambda a :(a[2]-a[0])*(a[3]-a[1])
    a_xmin,a_ymin,a_xmax,a_ymax=a
    b_xmin,b_ymin,b_xmax,b_ymax=b
    xminmax=max(a_xmin,b_xmin)
    xmaxmin=min(a_xmax,b_xmax)
    yminmax=max(a_ymin,b_ymin)
    ymaxmin=min(a_ymax,b_ymax)
    if xminmax>=xmaxmin or yminmax>=ymaxmin:
        return 0
    s1=calc_area(a)
    s2=calc_area(b)
    sc=(xmaxmin-xminmax)*(ymaxmin-yminmax)
    #return sc/(s1+s2-sc)
    return sc/min(s1,s2) # it is not actually IoU, but this is helpful when smaller one is coverd by the bigger one

def expand_bbox_scale(origin_bbox, target_size, image_size, expand_ratio=0.3):
    """
    Function:
    1. Moderately expand the original bounding box (padding).
    2. Ensure the expanded box stays within image boundaries.
    3. Scale box coordinates from the original image space to target size.

    Input:
        origin_bbox: [xmin, ymin, xmax, ymax]
        target_size: (target_w, target_h) target image size
        image_size: (image_w, image_h) original image size
        expand_ratio: expansion ratio, 0.1 means 10% context on width and height
    
    Output:
        [new_xmin, new_ymin, new_xmax, new_ymax] integer coordinates in target_size space
    """
    # Unpack coordinates and sizes.
    xmin, ymin, xmax, ymax = origin_bbox
    tw, th = target_size
    iw, ih = image_size

    # --- Step 1: Compute original box size ---
    box_w = xmax - xmin
    box_h = ymax - ymin

    # --- Step 2: Moderate expansion ---
    # Compute total padding and apply half on each side.
    # For simplicity, expand symmetrically by (w * ratio / 2).
    pad_w = int(box_w * expand_ratio / 2)
    pad_h = int(box_h * expand_ratio / 2)

    new_xmin = xmin - pad_w
    new_ymin = ymin - pad_h
    new_xmax = xmax + pad_w
    new_ymax = ymax + pad_h

    # --- Step 3: Clip to image boundaries ---
    # Ensure coordinates are within [0, image_size - 1].
    new_xmin = max(0, new_xmin)
    new_ymin = max(0, new_ymin)
    new_xmax = min(iw - 1, new_xmax) # Max valid index is length - 1.
    new_ymax = min(ih - 1, new_ymax)

    # return [new_xmin, new_ymin, new_xmax, new_ymax]

    # --- Step 4: Scale coordinates by ratio ---
    # Scale factor = target size / original size.
    scale_x = tw / iw
    scale_y = th / ih

    # Map coordinates from original space to target space.
    final_xmin = int(new_xmin * scale_x)
    final_ymin = int(new_ymin * scale_y)
    final_xmax = int(new_xmax * scale_x)
    final_ymax = int(new_ymax * scale_y)

    return [final_xmin, final_ymin, final_xmax, final_ymax]

def expand_bbox(origin_bbox,target_size,image_size):
    xmin,ymin,xmax,ymax=origin_bbox
    tw,th=target_size
    w,h=image_size
    if xmax-xmin+1>=tw or ymax-ymin+1>=th:
        t=max(xmax-xmin+1,ymax-ymin+1)
        tw=th=t
    #center
    all_expand=lambda xmn,xmx,l:[(xmn-l//2,xmx+(l-l//2)),(xmn-l,xmx),(xmn,xmx+l)]
    # all_expand=lambda xmn,xmx,l:[(xmn-l//2,xmx+(l-l//2)),(xmn-l,xmx),(xmn,xmx+l)]+([(0,xmx-xmn+l)] if l-xmn>0 else [])  # Handles elongated strips, but kept disabled for simplicity.
    check_bound=lambda xmn,xmx,l:xmn>=0 and xmx<l
    
    # If all three expansion strategies hit boundaries, fall back to full image.
    # @lyz: This may cause edge anomalies to become elongated strips in some cases.
    new_xmin,new_xmax=0,w-1
    new_ymin,new_ymax=0,h-1
    if xmax-xmin+1<tw:
        dl=tw-xmax+xmin-1  # delta_l: remaining width to reach target.
        for i in all_expand(xmin,xmax,dl):
            if check_bound(i[0],i[1],w):
                new_xmin,new_xmax=i
                break
    else:
        new_xmin,new_xmax=xmin,xmax
    
    if ymax-ymin+1<th:
        dl=th-ymax+ymin-1  # delta_l: remaining height to reach target.
        for i in all_expand(ymin,ymax,dl):
            if check_bound(i[0],i[1],h):
                new_ymin,new_ymax=i
                break
    else:
        new_ymin,new_ymax=ymin,ymax
    return [new_xmin,new_ymin,new_xmax,new_ymax]

def gen_random_bbox(patch_size,img_size,prob_mask=None): # img size (w,h)
    if type(patch_size)!=tuple and type(patch_size)!=list:
        patch_size=(patch_size,patch_size)
    patch_size=(min(patch_size[0],img_size[0]),min(patch_size[1],img_size[1]))
    eff_size=(img_size[0]-patch_size[0]+1,img_size[1]-patch_size[1]+1)
    w,h=eff_size
    hxw=h*w
    if prob_mask is None:
        prob_mask=np.ones((h,w))/hxw
    assert prob_mask.shape[0]==h and prob_mask.shape[1]==w, "prob mask shape error"
    idx_f=np.random.choice(hxw,p=prob_mask.flatten())
    xmin=idx_f%w
    ymin=idx_f//w
    return [xmin,ymin,xmin+patch_size[0]-1,ymin+patch_size[1]-1]

def make_bbox_ready(b):
    return [b[0],b[1],b[2]+1,b[3]+1]

def get_bboxes_from_mask(mask,target_size,w,h):
    cts=cv2.findContours(mask,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
    all_boxes=[]
    for c in cts[0]:
        xmin=c[...,0].min()
        xmax=c[...,0].max()
        ymin=c[...,1].min()
        ymax=c[...,1].max()
        op=[xmin,ymin,xmax,ymax]
        
        p=expand_bbox(op,(target_size,target_size),(w,h))# Use either expand_bbox or expand_bbox_scale.
        
        flg=True
        for j in all_boxes:
            if calc_IoU(op,j)>0.8:
                flg=False
                break
            if calc_IoU(p,j)>0.1:
                flg=False
                j[0]=min(j[0],p[0])
                j[1]=min(j[1],p[1])
                j[2]=max(j[2],p[2])
                j[3]=max(j[3],p[3])
                break
        
        # --- 3. Add as new box only if not merged and large enough ---
        box_w = p[2] - p[0]
        box_h = p[3] - p[1]
        if flg and box_w >= 7 and box_h >= 7:
            all_boxes.append(p)
    
    # 4. Final shape normalization via expand_bbox.
    all_boxes=[expand_bbox(b,(1,1),(w,h)) for b in all_boxes] # Make boxes square; can introduce overlap in some cases.
    
    if len(all_boxes) > 4:
        all_boxes=all_boxes[:4] # Should sort by score; current truncation is a temporary OOM-safe fallback.
    return all_boxes

def draw_and_crop_rois(image, all_boxes, second_image=None):
    """
    Input:
        image: source image
        all_boxes: list of bounding boxes
        second_image: (optional) second comparison image
    Output:
        img_list: 
            - Normal mode: [global-annotated image, crop1, crop2, ...]
            - One-shot mode: [global1, global2, stitched_crop1, stitched_crop2, ...]
    """

    # Font loading: prefer common serif fonts similar to Times New Roman, then fall back.
    font_size = 20
    font = None
    font_candidates = [
        "Times New Roman.ttf", "Times New Roman", "times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf", "LiberationSerif.ttf"
    ]
    for f in font_candidates:
        try:
            font = ImageFont.truetype(f, font_size)
            break
        except Exception:
            continue
    if font is None:
        # Final fallback to PIL default font.
        # print("Warning: preferred serif fonts not found. Using default font.")
        font = ImageFont.load_default()    

    # --- Internal helper: draw and crop for a single image ---
    def _process_single_img(target_img, boxes):
        target_np = np.array(target_img)
        if len(target_np.shape) == 3 and target_np.shape[2] == 3:
             img_bgr = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
        else:
             img_bgr = target_np.copy()

        crops = []
        img_h, img_w = target_np.shape[:2]

        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = map(int, box)
            
            # Draw red rectangle.
            cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            
            # Draw box index label.
            label_text = str(i + 1)
            text_x = xmin
            text_y = ymin - 5 if ymin >= 20 else ymin + 20
            cv2.putText(img_bgr, label_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Crop ROI.
            slice_ymin = max(0, ymin)
            slice_ymax = min(img_h, ymax + 1)
            slice_xmin = max(0, xmin)
            slice_xmax = min(img_w, xmax + 1)
            
            roi = target_np[slice_ymin:slice_ymax, slice_xmin:slice_xmax]
            if roi.size == 0:
                roi = np.zeros((10, 10, 3), dtype=np.uint8)
                
            crops.append(Image.fromarray(roi))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        global_res = Image.fromarray(img_rgb)
        
        return global_res, crops

    # ================= Main pipeline =================
    
    # 1. Process the first image.
    global_img1, crops1 = _process_single_img(image, all_boxes)

    final_list = []

    if second_image is None:
        # --- Mode A: standard mode ---
        final_list = [global_img1] + crops1
    else:
        # --- Mode B: one-shot comparison mode ---
        global_img2, crops2 = _process_single_img(second_image, all_boxes)
        
        final_list = [global_img1, global_img2]
        
        # Layout parameters.
        sep_width = 10         # Separator width.
        header_height = 35     # Header height (slightly increased for serif font readability).
        sep_color = (255, 255, 255) 
        line_color = (0, 0, 0)      
        text_color = (0, 0, 0)      
        
        # Use enumerate index j to map to box id (j+1).
        for j, (c1, c2) in enumerate(zip(crops1, crops2)):
            box_id = j + 1
            
            # --- 2.1 Prepare captions ---
            text1 = f"Image 1: bbox {box_id}"
            text2 = f"Image 2: bbox {box_id}"
            
            # --- 2.2 Create canvas ---
            total_width = c1.width + c2.width + sep_width
            max_height = max(c1.height, c2.height)
            canvas_height = max_height + header_height
            
            stitched_crop = Image.new('RGB', (total_width, canvas_height), sep_color)
            draw = ImageDraw.Draw(stitched_crop)
            
            # --- 2.3 Draw header text (PIL) ---
            
            # Compute text1 position (centered above image 1).
            # textbbox returns (left, top, right, bottom).
            bbox1 = draw.textbbox((0, 0), text1, font=font)
            w1 = bbox1[2] - bbox1[0]
            h1 = bbox1[3] - bbox1[1]
            x1 = (c1.width - w1) // 2
            y_text = (header_height - h1) // 2 - 2 # Small vertical centering adjustment.
            draw.text((max(0, x1), y_text), text1, fill=text_color, font=font)
            
            # Compute text2 position (centered above image 2).
            bbox2 = draw.textbbox((0, 0), text2, font=font)
            w2 = bbox2[2] - bbox2[0]
            # x = width(img1) + separator + centered offset on img2.
            x2 = c1.width + sep_width + (c2.width - w2) // 2
            draw.text((max(0, x2), y_text), text2, fill=text_color, font=font)
            
            # --- 2.4 Paste crops and draw separator ---
            stitched_crop.paste(c1, (0, header_height))
            stitched_crop.paste(c2, (c1.width + sep_width, header_height))
            
            line_x = c1.width + (sep_width // 2)
            draw.line([(line_x, 0), (line_x, canvas_height)], fill=line_color, width=2)
            
            final_list.append(stitched_crop)
    # ==========================================
    # Debug save logic (works for both modes).
    # ==========================================
    save_dir = "./output_tmp_roi"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save debug outputs with a simple safety check.
    if os.path.exists(save_dir):
        for i, img_item in enumerate(final_list):
            img_item.save(os.path.join(save_dir, f"debug_{i}.png"))
        # print(f"DEBUG: Images saved to {os.path.abspath(save_dir)}")
    # ==========================================

    return final_list

def prepare_inputs_with_masks(inputs: dict, one_shot=False):
    images=inputs.get("images",None)
    size_bd=inputs.get("size_bd",None) # Extra segmentation boundary for MVTec-LOCO ROI box drawing.
    # Add ROI crops and rectangle overlays.
    mask = inputs.get("masks", None)[0]
    bboxes = inputs.get("boxes_list", None)
    # images_list = self.process_image_with_mask(images[0], masks[0])
    max_box_num=4
    target_size=224 #384
    w,h=images[0].size
    shuffle_box=True

    if bboxes is not None:
        all_boxes=bboxes
    elif mask is None:
        all_boxes = [] # Test mode: do not generate boxes.
    else:# Convert mask to bounding boxes.
        if not isinstance(mask,np.ndarray):
            mask=np.array(mask)
            if mask.shape[0]!=h or mask.shape[1]!=w:
                mask=cv2.resize(mask,(w,h))
        else: #persume np array mask is normalized 0-1
            if mask.shape[0]!=h or mask.shape[1]!=w:
                mask=cv2.resize(mask,(w,h)) # should not use cv2 resize, but for now it is convenient
            mn_score,mx_score=mask.min(),mask.max()
            dscore=(mx_score-mn_score)*0.9+mn_score
            _,mask=cv2.threshold(mask,dscore,255,cv2.THRESH_BINARY)
            mask=mask.astype(np.uint8)
        
        # Three box-generation modes are available; choose by mode.
        if size_bd is not None:
            w = size_bd[2]
            h = size_bd[3]
        all_boxes=get_bboxes_from_mask(mask,target_size,w,h)

    # Shuffle boxes.
    if shuffle_box:
        np.random.shuffle(all_boxes)# Randomize box order.

    if len(all_boxes)>max_box_num:
        all_boxes=all_boxes[:max_box_num]
        print("remove extra boxes >",max_box_num)
    assert len(all_boxes)<=max_box_num, f"too many patches for now {len(all_boxes)}"
    
    # --- Core change: call draw_and_crop_rois ---
    if not one_shot:
        # 0-shot: pass only the first image.
        images_list = draw_and_crop_rois(images[0], all_boxes, second_image=None)
    else:
        # 1-shot: pass first and second images (images[1] is reference/comparison).
        assert len(images) >= 2, "One-shot mode requires at least 2 images."
        images_list = draw_and_crop_rois(images[0], all_boxes, second_image=images[1])
    
    inputs["boxes_list"] = all_boxes
    inputs["images_list"] = images_list
    return inputs

'''================================================= AGENT Modules ================================================='''
class BaseExpert(abc.ABC):
    @abc.abstractmethod
    def name(self):
        pass
    
    @abc.abstractmethod
    def description(self):
        pass

    @abc.abstractmethod
    def execute(self, query):
        pass

class ReferenceExtractor(BaseExpert):# Not tested yet.
    """0. ReferenceExtractor: 根据当前图像索引参考样本"""
    def __init__(self):
        pass

    def name(self):
        return "ReferenceExtractor"
    
    def description(self):
        return "根据query sample的图像特征搜索最接近的normal sample图像"
    
    def execute(self, task_info: dict):
        pass

class KnowledgeExpert(BaseExpert):
    """
    ✅✅✅
    KnowledgeExpert: 知识检索 (基于 domain_knowledge.json)
    根据传入的 dataset_type 过滤加载对应的领域知识。
    """
    def __init__(self, one_shot: bool=False, ds_cls: str="MVTec", knowledge_base_path="./data/domain_knowledge.json"):
        """
        初始化知识库专家。
        Args:
            ds_cls (str): 数据集类型名称 (例如: 'MVTec', 'VisA', 'GoodsAD')。
                           必须与 domain_knowledge.json 中的一级 Key 对应 (不区分大小写)。
            knowledge_base_path (str): 知识库 JSON 文件的路径。
        """
        self.one_shot = one_shot
        self.ds_cls = ds_cls
        self.kb_path = knowledge_base_path
        self.knowledge_map = self._load_and_process_knowledge()

    def name(self):
        return "KnowledgeExpert"

    def description(self):
        return f"加载 '{self.ds_cls}' 数据集的知识库，根据物体类别和缺陷关键词检索详细特征"

    def _load_and_process_knowledge(self):
        """
        加载 JSON 并仅提取指定 ds_cls 的数据，将其扁平化处理。
        结构转换为: { "category_name": { "defect_type": "description", ... } }
        """
        if not os.path.exists(self.kb_path):
            print(f"Warning: Knowledge base file not found at {self.kb_path}")
            return {}

        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f) #
            
            processed_data = {}
            
            # 1. If user requests all datasets (self.ds_cls == 'all'), merge all top-level keys.
            if isinstance(self.ds_cls, str) and self.ds_cls.lower() == "all":
                print("Loading knowledge for ALL datasets in the knowledge base.")
                for dataset_key, dataset_content in raw_data.items():
                    if not isinstance(dataset_content, dict):
                        continue
                    for category, defects in dataset_content.items():
                        cat_key = category.lower()
                        if cat_key not in processed_data:
                            processed_data[cat_key] = {}
                        if isinstance(defects, dict):
                            for defect_type, description in defects.items():
                                processed_data[cat_key][defect_type.lower()] = description
                return processed_data

            # Otherwise, find the matching dataset key (case-insensitive).
            # Example: user passes 'mvtec' while JSON uses 'MVTec'.
            target_key = None
            for key in raw_data.keys():
                if key.lower() == self.ds_cls.lower():
                    target_key = key
                    break

            if not target_key:
                print(f"Warning: Dataset type '{self.ds_cls}' not found in knowledge base.")
                return {}

            print(f"Loading knowledge for dataset: {target_key}")

            # 2. Load only categories under this dataset.
            dataset_content = raw_data[target_key] #

            # Iterate categories in this dataset (e.g., bottle, cable).
            for category, defects in dataset_content.items():
                # Normalize to lowercase for matching.
                cat_key = category.lower()

                if cat_key not in processed_data:
                    processed_data[cat_key] = {}

                # Iterate defect types for this category.
                for defect_type, description in defects.items():
                    # Store lowercase keys while keeping original descriptions.
                    processed_data[cat_key][defect_type.lower()] = description
            
            return processed_data
            
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return {}

    def execute(self, task_info: dict):
        """
        根据查询关键词检索当前已加载数据集中的知识。
        """
        if not self.knowledge_map:
            return f"Knowledge base for dataset '{self.ds_cls}' is empty or not loaded."

        query = task_info.get("obj_cls", "").lower()
        results = []
        
        # 1. Find object categories mentioned in query (within current dataset only).
        target_categories = [cat for cat in self.knowledge_map.keys() if cat in query]
        # Fallback: if no category matches, try fuzzy defect keyword matching.
        if not target_categories:
            
            generic_hits = []
            for cat, defects in self.knowledge_map.items():
                for d_type, desc in defects.items():
                    # Match defect keywords and ignore very short tokens.
                    if d_type in query and len(d_type) > 3: 
                        # Keep full description content without truncation.
                        generic_hits.append(f"【{cat} - {d_type}】: {desc}")
            
            if generic_hits:
                # Return concatenated lines directly; cap to top 3 matches.
                return "Potential Relevant Knowledge:\n" + "\n".join(generic_hits[:3])
            
            # Show supported categories for current dataset.
            supported_cats = ", ".join(list(self.knowledge_map.keys())[:5])
            return f"Category not found in '{self.ds_cls}' knowledge base. Supported categories include: {supported_cats}..."

        # 2. Perform deeper retrieval for matched categories.
        for cat in target_categories:
            defect_dict = self.knowledge_map[cat]
            if "good" in defect_dict:
                results.append(f"{defect_dict['good']}")
            found_specific_defect = False
            
            # Check whether a specific defect is mentioned.
            for d_type, desc in defect_dict.items():
                if d_type == "good":
                    continue
                normalized_dtype = d_type.replace("_", " ")
                if d_type in query or normalized_dtype in query:
                    results.append(f"{desc}")
                    found_specific_defect = True
            
        # 3. If category is found but no specific defect is mentioned, provide context.
        if not found_specific_defect:
            for d_type, desc in defect_dict.items():
                    if d_type != "good":
                        results.append(f"{desc}")

        return "\n".join(results)
    
class ReasonExpert(BaseExpert):
    '''✅✅✅'''
    def __init__(self, one_shot: bool=False, ds_cls: str="MVTec"):
        self.one_shot = one_shot
        self.ds_cls = ds_cls

    def name(self):
        return "ReasonExpert"
    
    def description(self):
        return "提供一个逻辑思考范式AD-CoT, Prompt文本形式 需要的LogicDescriber / KnowledgeExpert内容在DecisionMaker中加载"
    
    def execute(self, task_info: dict):
        """
        input: task_info 包含
        1. 异常检测类型 ad_cls 决定使用什么样的思考范式; 
        2. img_num 输入图像数量
        
        output: 提供模型推理的文本提示词
        """
        ad_cls = task_info.get("ad_cls", "structural") # "logical" "structural"
        img_num = task_info.get("img_num", 1)
        
        steps = []
        if self.one_shot and ad_cls=="logical": # 1-shot logical anomaly template (requires [LogicExpert] and [KnowledgeExpert]).
            return(
                "1. **Observe** the key logical characteristics (location, distribution, quantity, proportion, etc.) of different components in Image 1, primarily the characteristics described in [KnowledgeExpert].\n"
                "2. **Compare** the corresponding key logical characteristics in Image 1 and Image 2, focusing on any logical differences or inconsistencies. When these logical differences or inconsistencies deviate from the descriptions in [LogicExpert], the descriptions in [LogicExpert] shall prevail.\n"
                "3. **Decide** based on these differences which option best describes the condition of Image 1."
                )
        elif self.one_shot and ad_cls=="structural": # 1-shot structural anomaly template.
            return(
                "1. **Observe** the key characteristics (color, texture, shape, etc.) in Image 1, primarily the characteristics described in [KnowledgeExpert].\n"
                "2. **Compare** these characteristics to those in Image 2, focusing on any visible differences or inconsistencies.\n"
                "3. **Decide** based on these differences which option best describes the condition of Image 1."
                )
        elif ad_cls=="logical": # 0-shot logical anomaly template (requires [LogicExpert] and [KnowledgeExpert]).
            return(
                "1. **Observe** the key logical characteristics (location, distribution, quantity, proportion, etc.) of different components in Image 1, primarily the characteristics described in [KnowledgeExpert].\n"
                "2. **Compare** the key logical characteristics of different components in Image 1 with the **Normal Characteristics** in [KnowledgeExpert], focusing on any logical differences or inconsistencies. When these logical differences or inconsistencies deviate from the descriptions in [LogicExpert], the descriptions in [LogicExpert] shall prevail.\n"
                "3. **Decide** based on these differences which option best describes the condition of Image 1."
                )
        elif ad_cls=="structural": # 0-shot structural anomaly template.
            steps.append("**Global Observe** the key characteristics (color, texture, shape, etc.) in Image 1, primarily the characteristics described in [KnowledgeExpert].") # Step 1: always included.
            if img_num > 1:
                if img_num == 2:
                    steps.append(f"**Local Examine** the key characteristics in the local cropped image (Image 2), focusing on any obvious visible differences or irregularities (such as unexpected marks, structural deformations, or deviations from a standard appearance).") # Step 2: only when image count equals 2.
                else:
                    steps.append(f"**Local Examine** the key characteristics in the local cropped images (from Image 2 to Image {img_num}), focusing on any obvious visible differences or irregularities (such as unexpected marks, structural deformations, or deviations from a standard appearance).") # Step 2: only when image count is greater than 2.

            steps.append("**Decide** based on these differences which option best describes the condition of Image 1.") # Step 3.
            
            formatted_steps = [f"{i}. {step}" for i, step in enumerate(steps, 1)] # enumerate(..., 1) starts numbering from 1.
            return "\n".join(formatted_steps)
        else:
            print("[WARNING] --- 不存在此类检测类型!!!")
            return ""

class LogicExpert(BaseExpert):
    '''✅✅✅'''
    def __init__(self, one_shot: bool=False, ds_cls: str="MVTec-LOCO"):
        self.one_shot = one_shot
        self.ds_cls = ds_cls

    def name(self):
        return "LogicExpert"
    
    def description(self):
        return "逻辑异常检测专用 根据图像分割掩码描述不同类型对象的类型、位置、成分比例, 不区分0/1-shot"
    
    def execute(self, task_info: dict):
        # Analyze logical anomalies from segmentation masks using simple class-position-proportion comparison.
        catalogy = task_info.get("obj_cls", None) # breakfast_box juice_bottle pushpins screw_bag splicing_connectors
        good_summary = load_reference_model(f"./data/Reference-Logic/reference_model_{catalogy}_checked.json") # Load reference database.

        seg_path = task_info.get("seg_path", None)
        cfg = get_context_from_path(seg_path)
        # print(f"\nAnalyzing test image: {os.path.basename(seg_path)}")
        seg_onehot = read_seg_mask(seg_path, num_cls=cfg.get("num_cls", 3))
        current_stats = single_process_mask(seg_onehot, box=cfg.get("box", None), min_pixels=cfg.get("min_pixels", 1), mode_class=catalogy)

        results_list = analyze(current_stats, good_summary, cfg, strictness=3.0) # strictness=3.0 means 3-sigma range (~99.7% for normal data).
        output = describe(results_list, cfg)
        # print(output)
        return output

class DecisionMaker(BaseExpert):
    '''✅✅✅'''
    def __init__(self, one_shot: bool=False, ds_cls: str="MVTec"):
        self.one_shot = one_shot
        self.ds_cls = ds_cls

    def name(self):
        return "DecisionMaker"
    
    def description(self):
        return "汇总 LogicExpert,KnowledgeExpert,ReasonExpert 等模块生成的提示词, 生成最终的提示词"
    
    def execute(self, task_info: dict, prompt_info: dict):

        ad_cls = task_info.get("ad_cls", "structural") # "logical" "structural"
        img_num = task_info.get("img_num", 0) # Number of images.
        prompt_know = prompt_info.get("KnowledgeExpert", None)
        prompt_desc = prompt_info.get("LogicExpert", None)
        prompt_reason = prompt_info.get("ReasonExpert", None)
        
        info_step = []
        if self.one_shot and ad_cls=="logical":
            # 1-shot logical anomaly prompt composition.
            # Image description.
            if img_num > 3: # Includes query sample, reference sample, and multiple cropped images.
                image_prefix = "<image>" * (img_num-2)
                info_step.append(f"Image 1 <image> is the query sample, where numbered red rectangular boxes mark regions of interest. Image 2 <image> is the normal reference sample, which has the same red rectangular boxes marking as Image 1. The other cropped images {image_prefix} correspond to these numbered boxes in sequential order (e.g., bbox 1, bbox 2...), each image presents a side-by-side comparison of these cropped regions (left from Image 1, right from Image 2). Use these images and the following information to help answer the question about Image 1.\n- Note: Focus only on logical anomalies in the regions of interest.")
            elif img_num == 3: # Includes query sample, reference sample, and one cropped image.
                info_step.append("Image 1 <image> is the query sample, where a numbered red rectangular box marks the region of interest. Image 2 <image> is the normal reference sample, which has the same red rectangular box markings as Image 1. The cropped image <image> presents a side-by-side comparison of the rectangular box regions (left from Image 1, right from Image 2). Use these images and the following information to help answer the question about Image 1.\n- Note: Focus only on logical anomalies in the regions of interest.")
            else:
                info_step.append("Image 1 <image> is the query sample and Image 2 <image> is the normal reference sample. Use these images and the following information to help answer the question about Image 1.\n- Note: Focus only on logical anomalies in Image 1.")
            
            if "know" in prompt_info["task_routing"]: # Knowledge block is optional.
                info_step.append(f"Following is the **domain knowledge** provided by [KnowledgeExpert], which contains all the possible types of defect characteristics:\n{prompt_know}")
                
            info_step.append(f"Following is the **key logical characteristics description** in Image 1 provided by [LogicExpert]:\n{prompt_desc}") # Description block is required.
            
            if "reason" in prompt_info["task_routing"]: # Reasoning block is optional.
                info_step.append(f"Follow the reasoning steps to approach the question systematically:\n{prompt_reason}")
            
        elif self.one_shot and ad_cls=="structural":
            # 1-shot structural anomaly prompt composition.
            # Image description.
            if img_num > 3: # Includes query sample, reference sample, and multiple cropped images.
                image_prefix = "<image>" * (img_num-2)
                info_step.append(f"Image 1 <image> is the query sample, where numbered red rectangular boxes mark potential anomaly regions requiring special attention. Image 2 <image> is the normal reference sample, which has the same red rectangular boxes marking as Image 1. The other cropped images {image_prefix} correspond to these numbered boxes in sequential order (e.g., bbox 1, bbox 2...), each image presents a side-by-side comparison of these cropped regions (left from Image 1, right from Image 2). Use these images and the following information to help answer the question about Image 1.")
            elif img_num == 3: # Includes query sample, reference sample, and one cropped image.
                info_step.append(f"Image 1 <image> is the query sample, where a numbered red rectangular box marks potential anomaly region requiring special attention. Image 2 <image> is the normal reference sample, which has the same red rectangular box marking as Image 1. The cropped image <image> presents a side-by-side comparison of the rectangular regions (left from Image 1, right from Image 2). Use these images and the following information to help answer the question about Image 1.")
            else:
                info_step.append("Image 1 <image> is the query sample and Image 2 <image> is the normal reference sample. Use these images and the following information to help answer the question about Image 1.")

            if "know" in prompt_info["task_routing"]: # Knowledge block is optional.
                info_step.append(f"Following is the **domain knowledge** provided by [KnowledgeExpert], which contains all the possible types of defect characteristics:\n{prompt_know}")
                
            if "reason" in prompt_info["task_routing"]: # Reasoning block is optional.
                info_step.append(f"Follow the reasoning steps to approach the question systematically:\n{prompt_reason}")
            
        elif ad_cls=="logical":
            # 0-shot logical anomaly prompt composition.
            # Image description.
            if img_num > 2: # Includes query sample and multiple cropped images.
                image_prefix = "<image>" * (img_num-1)
                info_step.append(f"Image 1 <image> is the query sample, where numbered red rectangular boxes mark regions of interest. The other cropped images {image_prefix} correspond to the details within these numbered boxes in sequential order (e.g., bbox 1, bbox 2...). Use these images and the following information to help answer the question about Image 1.\n- Note: Focus only on logical anomalies in the regions of interest.")
            elif img_num == 2: # Includes query sample and one cropped image.
                info_step.append("Image 1 <image> is the query sample, where a numbered red rectangular box mark the region of interest. The cropped image <image> correspond to the details within the rectangular box. Use these images and the following information to help answer the question about Image 1.\n- Note: Focus only on logical anomalies in the region of interest.")
            else:
                info_step.append("Image 1 <image> is the query sample. Use it and the following information to help answer the question about Image 1.\n- Note: Focus only on logical anomalies in Image 1.")
            
            if "know" in prompt_info["task_routing"]: # know可选
                info_step.append(f"Following is the **domain knowledge** provided by [KnowledgeExpert], which contains all the possible types of defect characteristics:\n{prompt_know}")
                
            info_step.append(f"Following is the **key logical characteristics description** in Image 1 provided by [LogicExpert]:\n{prompt_desc}") # desc必选

            if "reason" in prompt_info["task_routing"]: # reason可选
                info_step.append(f"Follow the reasoning steps to approach the question systematically:\n{prompt_reason}")

        elif ad_cls=="structural":
            # 0-shot结构异常提示词 组合
            # 图像描述
            if img_num > 2: # 包含 query sample 和多张 cropped images
                image_prefix = "<image>" * (img_num-1)
                info_step.append(f"Image 1 <image> is the query sample, where numbered red rectangular boxes mark potential anomaly regions requiring special attention. The other cropped images {image_prefix} correspond to the details within these numbered boxes in sequential order (e.g., bbox 1, bbox 2...). Use these images and the following information to help answer the question about Image 1.")
            elif img_num == 2: # 包含 query sample 和一张 cropped image
                info_step.append("Image 1 <image> is the query sample, where numbered red rectangular boxes mark potential anomaly regions requiring special attention. The cropped image <image> correspond to the details within the rectangular box. Use these images and the following information to help answer the question about Image 1.")
            else:
                info_step.append("Image 1 <image> is the query sample. Use it and the following information to help answer the question about Image 1.")
            
            if "know" in prompt_info["task_routing"]: # know可选
                info_step.append(f"Following is the **domain knowledge** provided by [KnowledgeExpert], which contains all the possible types of defect characteristics:\n{prompt_know}")
                
            if "reason" in prompt_info["task_routing"]: # reason可选
                info_step.append(f"Follow the reasoning steps to approach the question systematically:\n{prompt_reason}")

        else:
            print("[WARNING] --- 不存在此类检测类型!!!")

        # info_step.append("Please respond with the letter of the correct option only.") # 必选
        total_prompt = "\n\n".join(info_step)
        return total_prompt

'''================================ 仅生成多模态Prompts 不加载模型 ======================================'''
class ExpertPromptsGenerator:
    def __init__(self, **kwargs):
        self.model_path = kwargs.get('model_path', None)
        self.teacher_type = kwargs.get("teacher_type", "API") # teacher模型采用 API 还是 本地模型
        if self.teacher_type == "API":
            # 获取 API Key 和 Base URL，通常从 kwargs 中获取，也可以硬编码
            api_key = kwargs.get('api_key', "EMPTY") 
            base_url = kwargs.get('base_url', "https://api.siliconflow.cn/v1") # 示例地址，需根据实际服务商修改
            
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.device = kwargs.get('device', 'cuda')
            device_map = self.device if self.device is not None else "auto"
            
            from transformers import AutoProcessor
            from transformers import Qwen2VLForConditionalGeneration
            from transformers import Qwen3VLForConditionalGeneration

            if "Qwen2-VL" in self.model_path:
                self.model_type = 'qwen2'
                model_class = Qwen2VLForConditionalGeneration
                # 这个里面把device_map和device合二为一了。
                self.model = model_class.from_pretrained(
                    self.model_path, torch_dtype="auto", device_map=device_map
                ) # dtype=torch.float16 or "auto"
                self.model.eval()
            elif 'Qwen3-VL' in self.model_path:# 已测试
                self.model_type = 'qwen3'
                model_class = Qwen3VLForConditionalGeneration
                self.model = model_class.from_pretrained(
                    self.model_path, dtype="auto", device_map=device_map
                )
                self.model.eval()
                
            else:
                raise NotImplementedError(f"不支持多模态模型: {self.model_path}")
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        self.one_shot = kwargs.get("one_shot", False) # 是否开启 one_shot
        self.ds_cls = kwargs.get("ds_cls", "MVTec") # 数据集名称

        # 初始化所有专家实例
        self.ref_extractor = ReferenceExtractor()
        self.knowledge_expert = KnowledgeExpert(one_shot=self.one_shot, ds_cls=self.ds_cls)
        self.logic_expert = LogicExpert(one_shot=self.one_shot, ds_cls=self.ds_cls)
        self.reason_expert = ReasonExpert(one_shot=self.one_shot, ds_cls=self.ds_cls)
        self.decision_maker = DecisionMaker(one_shot=self.one_shot, ds_cls=self.ds_cls)

        # 定义任务路由表 后续根据 self.one_shot与ad_cls选择性补充
        self.task_routing = {
            "Anomaly Detection": ["decision"],
            "Defect Classification": ["know", "decision"],
            "Defect Localization": ["decision"],
            "Defect Description": ["know", "reason", "decision"],
            "Defect Analysis": ["know", "reason", "decision"],
            
            "Object Classification": ["know", "decision"],
            "Object Structure": ["decision"],
            "Object Details": ["decision"],
            "Object Analysis": ["decision"],
        }

    def _plan_task(self, task_type: str, ad_cls: str) -> str:
        """
        Decision Planning: 根据任务类型与异常类型，分配专家模块
        """
        origin_experts = self.task_routing.get(task_type, ["decision"])
        active_experts = origin_experts.copy()
        if ad_cls == "logical":
            active_experts.insert(0, "mask2text")#!!!!!!!!!需要修改
            if "know" not in active_experts:
                active_experts.insert(1, "know")
        
        return active_experts # 默认 fallback

    def _pil_to_base64(self, image):
        """将 PIL 图片转换为 Base64 字符串"""
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _build_api_messages(self, state):
        """
        将内部状态 (state['state'] + state['images']) 转换为 API 消息格式
        """
        messages = []
        all_images = state.get('images', [])
        image_idx = 0
        
        for turn in state['state']:
            role = turn['role']
            content = turn['content']
            
            new_content = []
            if isinstance(content, str):
                new_content = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            new_content.append({"type": "text", "text": item['text']})
                        elif item.get('type') == 'image':
                            if image_idx < len(all_images):
                                base64_img = self._pil_to_base64(all_images[image_idx])
                                new_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                                })
                                image_idx += 1
            
            messages.append({"role": role, "content": new_content})
            
        return messages

    def _add_text(self, state, text, new_images=None):
        assert isinstance(new_images, list), "输入的图像应该是一个队列List."
        if self.teacher_type == "API":           
            # 保持原有的内部状态结构
            text_parts = text.split("<image>")
            content_list = []
            for i, part in enumerate(text_parts):
                if part: # 防止空字符串
                    content_list.append({"type": "text", "text": part})
                if i < len(new_images):
                    content_list.append({"type": "image",})
            state['state'].append({
                "role": "user",
                "content": content_list
            })
            state['images'] += new_images
        else:
            # 保持原有的内部状态结构
            text_parts = text.split("<image>")
            content_list = []
            for i, part in enumerate(text_parts):
                if part: # 防止空字符串
                    content_list.append({"type": "text", "text": part})
                if i < len(new_images):
                    content_list.append({"type": "image", "image": new_images[i]})
            state['state'].append({
                "role": "user",
                "content": content_list
            })
        return state
    
    def _new_state(self):
        return {"state": [], "images": []}

    def process(self, state=None, task_info:dict={}): 
        if state is None:
            state = self._new_state()
        
        query = task_info.get("query", None)
        task_type = task_info.get("task_type", "Anomaly Detection")
        ad_cls = task_info.get("ad_cls", "structural")
        
        # 1. 规划 (Planning)
        active_experts = self._plan_task(task_type, ad_cls)   # 根据关键词确定任务类别
        
        if self.one_shot:
            active_experts.insert(0, "ref")
        
        # print(f"[-] 检测到任务类型: {task_type}")
        # print(f"[-] 激活专家模块: {active_experts}")
        
        context_buffer = {} # 根据任务类型构建一个context_buffer
        context_buffer["task_routing"] = active_experts
        # 2. 选择性执行专家模块 (Execution Pipeline)
        if "ref" in active_experts:
            # 此处为伪代码 将查询得到的图像进行处理后 返回PLI图像列表
            context_buffer["ReferenceExtractor"] = self.ref_extractor.execute(task_info)
        if "mask2text" in active_experts:
            # 此处为伪代码 将逻辑异常包含的掩码转换成固定格式的逻辑描述 返回相关提示词
            context_buffer["LogicExpert"] = self.logic_expert.execute(task_info)
        if "know" in active_experts:    # [Expert: Knowledge Expert]
            context_buffer["KnowledgeExpert"] = self.knowledge_expert.execute(task_info)
        if "reason" in active_experts:  # [Expert: Reason Expert]
            context_buffer["ReasonExpert"] = self.reason_expert.execute(task_info)
        
        expert_prompt = self.decision_maker.execute(task_info, context_buffer)
        # print(expert_prompt)
        return expert_prompt
    
    def teacher_generate(self, new_text, images=None, state=None,  **generate_kwargs):
        # 包装最终的多模态prompts 调用模型生成COT回答
        if state is None:
            state = self._new_state()
        if new_text is not None:
            state = self._add_text(state, new_text, new_images=images)

        if self.teacher_type == "API":
            # 准备 API 参数
            messages = self._build_api_messages(state)
            
            # 转换生成参数 (映射 transformers 参数到 openai 参数)
            api_kwargs = {
                "model": self.model_path,
                "messages": messages,
                "temperature": generate_kwargs.get("temperature", 0.7),
                "max_tokens": generate_kwargs.get("max_new_tokens", 512),
                "top_p": generate_kwargs.get("top_p", 0.7),
            }
            
            # 重试逻辑：最多尝试 5 次，每次失败等待 3 秒后重试
            max_attempts = 5
            delay_seconds = 3
            attempt = 0
            output_text = None
            while attempt < max_attempts:
                try:
                    response = self.client.chat.completions.create(**api_kwargs)
                    output_text = response.choices[0].message.content
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        output_text = f"Error calling API after {attempt} attempts: {str(e)}"
                        print(output_text)
                        break
                    else:
                        print(f"API call failed (attempt {attempt}/{max_attempts}), retrying in {delay_seconds}s: {e}")
                        time.sleep(delay_seconds)

            # 将结果写回状态，保持一致性
            state['state'].append({
                'role': 'assistant', 
                'content': output_text,
            })
                
            return state, output_text
        else:
            inputs = self.processor.apply_chat_template(
                    state['state'],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
            inputs = inputs.to(self.device)
            # Inference: Generation of the output
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generate_kwargs)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            state['state'].append({
                'role': 'assistant', 
                'content': [
                    {
                        'type': 'text', 
                        'text': output_text[0]
                    }
                ]  # <--- 封装成列表 + 字典的标准格式
            })
            return state, output_text[0]
    
    def single_expert_gen(self, inputs, state=None):# 增加了ROI图像区域分割引导
        images_list = inputs.get("images_list", None)
        ad_cls = inputs.get("ad_class", "structural") # 逻辑异常问题不裁剪ROI区域
        # if ad_cls == "logical" and not self.one_shot and images_list:
        #     images_list = images_list[:1] # 逻辑异常0-shot只传入第一张图

        task_info = {
            "query": inputs.get("text",None),
            "img_num": len(images_list),
            "task_type": inputs.get("task_type", None),
            "image_path": inputs.get("image_path", None),
            "seg_path": inputs.get("seg_path", None),
            "ad_cls": ad_cls,
            "obj_cls": inputs.get("obj_class", None),
        }
        
        return self.process(state=state, task_info=task_info)
    
    def multi_qa_gen(self, inputs, convs, forget_state=False, **generate_kwargs): # maintain train data formation
        generate_kwargs = {
            "temperature": inputs.get("temperature",0.2),
            "top_p":  inputs.get("top_p",0.7),
            "max_new_tokens": inputs.get("max_new_tokens",512),
        }
        images_list = inputs.get("images_list", None)
        state=self._new_state()
        
        for conv_idx in range(0,len(convs),2):
            cur_qs = convs[conv_idx]["value"].strip()
            inputs["text"] = cur_qs
            inputs["task_type"] = inputs["task_types"][conv_idx // 2]
            
            if forget_state:
                state = self._new_state()
            expert_prompt = self.single_expert_gen(inputs, state)
            user_prompt = f"# **Instructions for Answering the Question**\n{expert_prompt}\n\n# **Question & Options**\n{cur_qs}"
            # image_prefix = "<image>\n" * img_num
            # user_prompt = image_prefix + user_prompt
            # print(user_prompt)
            cur_ans_gt = convs[conv_idx + 1]["value"].strip()
            
            # 构造 User Prompt：要求教师基于 student_instruction 进行思考
            prompt_text = f"""You are an expert AI Tutor.
Your task is to generate a training response that teaches a student model **how to derive the answer** by combining the `<instruction>` with visual observations.

# **Crucial Context:**
- **The Student Model ONLY sees:** The Images and the `<instruction>`. It does NOT know the answer.
- **You (The Teacher) see:** The `<ground_truth>` (the correct answer).
- **Your Goal:** Write a "Chain-of-Thought" that simulates a perfect student's reasoning process: starting from the instruction, observing the image, and **logically concluding** with the answer that matches the `<ground_truth>`.

# **Core Logic (The Inference Path):**
1. **Start with the Instruction**: Acknowledge the context or guide provided in `<instruction>`. (e.g., "The instruction guides me to focus on...")
2. **Observe the Images**: Describe the visual details visible in the images. **This is the most important part.** You must describe the specific features that lead to the conclusion.
3. **Synthesize & Conclude**: Explain how the visual evidence matches (or contradicts) the instruction's information to result in the final conclusion.
  - *Note: You are aiming for the `<ground_truth>`, but your reasoning must sound like it is BEING FOUND, not already known.*

# **Strict Output Format:** Do NOT use markdown backticks (```) or code blocks. Output the response strictly using the following XML-style tags:
<reason>
The step-by-step deduction. Start with the instruction, describe **key** visual features in the image/ROI, cite the rule, and derive the conclusion. **Keep it concise (max 4-5 sentences) and avoid filler words.** NEVER mention that you have access to the Ground Truth; act as if you are discovering it.
</reason>
<answer>
The concise final conclusion matching the Ground Truth.
</answer>

# **Input Data:**
<instruction>
{expert_prompt}
</instruction>

<question>
{cur_qs}
</question>

<ground_truth>
{cur_ans_gt}
</ground_truth>"""

# The answer is: [Content from `<ground_truth>`].
# [Reasoning about answering questions, summarize in a single paragraph.]
# Output ONLY a valid JSON object (no markdown backticks). The JSON must have exactly two keys:
# The step-by-step deduction. Start with the instruction, describe specific visual features in the image/ROI, cite the rule, and derive the conclusion. NEVER mention that you have access to the Ground Truth; act as if you are discovering it.
# Your reasoning text here.
# Reasoning: A logical derivation paragraph. Step 1: Read Instruction -> Step 2: Analyze Image -> Step 3: Draw Conclusion.
            # print(f"\n[Teacher Prompt]:\n{prompt_text}\n")
            convs[conv_idx]["old_value"] = convs[conv_idx]["value"]
            convs[conv_idx]["value"] = user_prompt
            state, out = self.teacher_generate(new_text=prompt_text, images=images_list, state=state, **generate_kwargs)
            convs[conv_idx + 1]["old_value"] = cur_ans_gt
            convs[conv_idx + 1]["value"] = out.strip()
        return convs

'''================================================= LMM模型调用类 ================================================='''
class CoExpert:
    """
    基于Qwen3VL的LMM AGENT框架
    """
    def __init__(self, model_path, task='LMM', **kwargs):
        
        self.one_shot = kwargs.get("one_shot", False) # 是否开启 one_shot
        self.ds_cls = kwargs.get("ds_cls", "MVTec") # 数据集名称

        # 初始化所有专家实例
        self.ref_extractor = ReferenceExtractor()
        self.knowledge_expert = KnowledgeExpert(one_shot=self.one_shot, ds_cls=self.ds_cls)
        self.logic_expert = LogicExpert(one_shot=self.one_shot, ds_cls=self.ds_cls)
        self.reason_expert = ReasonExpert(one_shot=self.one_shot, ds_cls=self.ds_cls)
        self.decision_maker = DecisionMaker(one_shot=self.one_shot, ds_cls=self.ds_cls)

        # 定义任务路由表 后续根据 self.one_shot与ad_cls选择性补充
        self.task_routing = {
            "Anomaly Detection": ["decision"],
            "Defect Classification": ["know", "decision"],
            "Defect Localization": ["decision"],
            "Defect Description": ["know", "reason", "decision"],
            "Defect Analysis": ["know", "reason", "decision"],
            
            "Object Classification": ["know", "decision"],
            "Object Structure": ["decision"],
            "Object Details": ["decision"],
            "Object Analysis": ["decision"],
        }
        
        self.task = task
        assert task in ['LLM', 'LMM'], f"CoExpert不支持{task}类型."
        self.device = kwargs.get('device', None)
        if self.task == 'LLM':
            from transformers import pipeline
            if self.device is not None:
                self.model = pipeline("text-generation", model_path, torch_dtype="auto", device=self.device)
            else:
                self.model = pipeline("text-generation", model_path, torch_dtype="auto", device_map="auto")
        if self.task == 'LMM':
            from transformers import AutoTokenizer, AutoProcessor # 临时注销 不加载模型
            from transformers import Qwen3VLForConditionalGeneration # 临时注销 不加载模型
            if self.device is not None:
                device_map = self.device
            else:
                device_map = "auto"

            if 'Qwen3-VL' in model_path:
                self.model_type = 'qwen3'
                # 临时注销 不加载模型
                model_class = Qwen3VLForConditionalGeneration
                # 这个里面把device_map和device合二为一了。
                self.model = model_class.from_pretrained(
                    model_path, dtype=torch.bfloat16, device_map=device_map,
                    attn_implementation="flash_attention_2",
                ) 
                self.model.eval()
                
            else:
                raise NotImplementedError(f"不支持多模态模型: {model_path}")
            # 临时注销 不加载模型
            self.processor = AutoProcessor.from_pretrained(model_path)# 用来把文本和图像预处理成模型需要的张量，并负责后处理
            if kwargs.get("adapter_path", None) is not None:
                adapter_path = kwargs["adapter_path"]
                print("正在加载 LoRA 适配器...")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)

    def _plan_task(self, task_type: str, ad_cls: str) -> str:
        """
        Decision Planning: 根据任务类型与异常类型，分配专家模块
        """
        origin_experts = self.task_routing.get(task_type, ["decision"])
        active_experts = origin_experts.copy()
        if ad_cls == "logical":
            active_experts.insert(0, "mask2text")#!!!!!!!!!需要修改
            if "know" not in active_experts:
                active_experts.insert(1, "know")
        
        return active_experts # 默认 fallback

    def _add_text(self, state, text, new_images=None):
        if self.task == 'LLM':
            assert new_images is None, "语言模型不支持输入图像."
            state.append({"role": "user", "content": text})
            return state
        elif self.task == 'LMM':
            if new_images is not None:
                assert isinstance(new_images, list), "输入的图像应该是一个队列List."
                if self.model_type == 'qwen3':
                    text_parts = text.split("<image>")
                    content_list = []
                    for i, part in enumerate(text_parts):
                        if part: # 防止空字符串
                            content_list.append({"type": "text", "text": part})
                        if i < len(new_images):
                            content_list.append({"type": "image", "image": new_images[i]})
                    state['state'].append({
                        "role": "user",
                        "content": content_list
                    })
            else:# 没有图像输入情况
                state['state'].append({
                    'role': 'user', 
                    'content': [
                        {
                            'type': 'text', 
                            'text': text
                        }
                    ]  # <--- 封装成列表 + 字典的标准格式
                })                  
            return state
        else:
            raise NotImplementedError(f"不支持{self.task}的任务类型")
    
    def _new_state(self):
        if self.task == 'LLM':
            return []
        elif self.task == 'LMM':
            return {
                "state": [], 
                "images": []
            }
        else:
            raise NotImplementedError(f"不支持{self.task}的任务类型")

    def generate(self, new_text, images=None, state=None,  **generate_kwargs):
        if state is None:
            state = self._new_state()
        if new_text is not None:
            state = self._add_text(state, new_text, new_images=images)

        # state = self._add_text(state, new_text, new_images=images)
        if self.task == 'LLM':
            response_message = self.model(state, **generate_kwargs)[0]["generated_text"][-1]
            state.append(response_message)
            return state, response_message['content']
        elif self.task == 'LMM':
            if self.model_type == "qwen3":
                inputs = self.processor.apply_chat_template(
                    state['state'],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                # Inference: Generation of the output
                generated_ids = self.model.generate(**inputs, **generate_kwargs)
                # generated_ids = self.model.generate(**inputs, tokenizer=self.tokenizer, **generate_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                state['state'].append({
                    'role': 'assistant', 
                    'content': [
                        {
                            'type': 'text', 
                            'text': output_text[0]
                        }
                    ]  # <--- 封装成列表 + 字典的标准格式
                })
                return state, output_text[0]
            
            else:# 其他模型 还未测试
                answer = self.model.chat(
                    image=None,
                    msgs=state['state'], 
                    tokenizer=self.tokenizer,
                    **generate_kwargs # 确保 kwargs 能传进去
                )
                state['state'].append({
                    'role': 'assistant', 
                    'content': [answer]
                })
                # print(answer)
                return state, answer

        else:
            raise NotImplementedError(f"不支持{self.task}的任务类型")

    def process(self, images=None, state=None, task_info:dict={}, **generate_kwargs): 
        if state is None:
            state = self._new_state()
        
        query = task_info.get("query", None)
        task_type = task_info.get("task_type", "Anomaly Detection")
        ad_cls = task_info.get("ad_cls", "structural")
        
        # 1. 规划 (Planning)
        active_experts = self._plan_task(task_type, ad_cls)   # 根据关键词确定任务类别
        
        if self.one_shot:
            active_experts.insert(0, "ref")
        
        # print(f"[-] 检测到任务类型: {task_type}")
        # print(f"[-] 激活专家模块: {active_experts}")
        
        context_buffer = {} # 根据任务类型构建一个context_buffer
        context_buffer["task_routing"] = active_experts
        # 2. 选择性执行专家模块 (Execution Pipeline)
        if "ref" in active_experts:
            # 此处为伪代码 将查询得到的图像进行处理后 返回PLI图像列表
            context_buffer["ReferenceExtractor"] = self.ref_extractor.execute(task_info)
        if "mask2text" in active_experts:
            # 此处为伪代码 将逻辑异常包含的掩码转换成固定格式的逻辑描述 返回相关提示词
            context_buffer["LogicExpert"] = self.logic_expert.execute(task_info)
        if "know" in active_experts:    # [Expert: Knowledge Expert]
            context_buffer["KnowledgeExpert"] = self.knowledge_expert.execute(task_info)
        if "reason" in active_experts:  # [Expert: Reason Expert]
            context_buffer["ReasonExpert"] = self.reason_expert.execute(task_info)
        
        sys_prompt = self.decision_maker.execute(task_info, context_buffer)
        user_prompt = f"# **Instructions for Answering the Question**\n{sys_prompt}\n\n# **Question & Options**\n{query}" # 把问题放到引导的最后
        # print(user_prompt)
        return self.generate(new_text=user_prompt, images=images, state=state, **generate_kwargs)  # 根据初始的信息输出生成理据
    
    def single_qa(self, inputs, state=None):# 增加了ROI图像区域分割引导
        images_list = inputs.get("images_list", None)
        ad_cls = inputs.get("ad_class", "structural")
        generate_kwargs = {
            "temperature": inputs.get("temperature",0.2),
            "top_p":  inputs.get("top_p",0.7),
            "max_new_tokens": inputs.get("max_new_tokens",512),
        }

        task_info = {
            "query": inputs.get("text",None),
            "img_num": len(images_list) if images_list is not None else 0,
            "task_type": inputs.get("task_type", None),
            "image_path": inputs.get("image_path", None),
            "seg_path": inputs.get("seg_path", None),
            "ad_cls": ad_cls,
            "obj_cls": inputs.get("obj_class", None),
        }
        return self.process(
            images=images_list, 
            state=state, 
            task_info=task_info,
            **generate_kwargs)
    
    def multi_qa(self,inputs,convs,forget_state=False): # maintain train data formation
        state=self._new_state()
        for conv_idx in range(0,len(convs),2):
            cur_qs=convs[conv_idx]["value"].strip()
            inputs["text"]=cur_qs
            inputs["task_type"] = inputs["task_types"][conv_idx // 2]
            
            if forget_state:
                state=self._new_state()
            state,out=self.single_qa(inputs,state)
            convs[conv_idx+1]["old_value"]=convs[conv_idx+1]["value"]
            convs[conv_idx+1]["value"]=out
        return convs

class TransformersModelWrapper:
    """
    Refer to [
        https://qwen.readthedocs.io/en/latest/inference/chat.html.  (LLM)
        https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct  (Qwen2VL)
        https://huggingface.co/docs/transformers/main/en/model_doc/llava_onevision#single-image-inference  (LLAVA-OneVision)
        https://huggingface.co/openbmb/MiniCPM-V-2_6  (MiniCPM-V 2.6)
    ]
    """
    def __init__(self, model_path, task='LMM', **kwargs):
        self.task = task
        assert task in ['LLM', 'LMM'], f"TransformersModelWrapper不支持{task}类型."
        self.device = kwargs.get('device', None)
        if self.task == 'LLM':
            from transformers import pipeline           
            if self.device is not None:
                self.model = pipeline("text-generation", model_path, torch_dtype="auto", device=self.device)
            else:
                self.model = pipeline("text-generation", model_path, torch_dtype="auto", device_map="auto")

        if self.task == 'LMM':
            from transformers import AutoTokenizer, AutoProcessor, AutoModel
            from transformers import Qwen2VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration
            from transformers import Qwen3VLForConditionalGeneration

            if self.device is not None:
                device_map = self.device
            else:
                device_map = "auto"

            if "Qwen2-VL" in model_path:
                self.model_type = 'qwen2'
                model_class = Qwen2VLForConditionalGeneration
                # 这个里面把device_map和device合二为一了。
                self.model = model_class.from_pretrained(
                    model_path, torch_dtype="auto", device_map=device_map
                )
                
            elif 'lmms-lab--llava-onevision' in model_path and '-hf' in model_path:  # 这种方式只能调用hf格式的llava-onevision模型。
                self.model_type = 'llava-onevision'
                model_class = LlavaOnevisionForConditionalGeneration
                # 这个里面把device_map和device合二为一了。
                self.model = model_class.from_pretrained(
                    model_path, torch_dtype="auto", device_map=device_map
                )
                self.model.eval()
            elif 'Qwen3-VL' in model_path:# 已测试
                self.model_type = 'qwen3'
                model_class = Qwen3VLForConditionalGeneration
                # 这个里面把device_map和device合二为一了。
                self.model = model_class.from_pretrained(
                    model_path, dtype=torch.bfloat16, device_map=device_map,
                    attn_implementation="flash_attention_2"
                )
                self.model.eval()           
            elif 'MiniCPM-V' in model_path:
                self.model_type = 'minicpm-v'
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(
                    model_path, trust_remote_code=True,
                    attn_implementation='sdpa', torch_dtype=torch.float16
                )
                self.model.eval()
                if self.device is not None:
                    self.model.to(self.device)
                
            else:
                raise NotImplementedError(f"不支持多模态模型: {model_path}")
            
            self.processor = AutoProcessor.from_pretrained(model_path)
    
    def _add_text(self, state, text, new_images=None):
        if self.task == 'LLM':
            assert new_images is None, "语言模型不支持输入图像."
            state.append({"role": "user", "content": text})
            return state
        elif self.task == 'LMM':
            if new_images is not None:
                assert isinstance(new_images, list), "输入的图像应该是一个队列List."
                if self.model_type == 'qwen3': # qwen3已测试
                    text_parts = text.split("<image>")
                    content_list = []
                    for i, part in enumerate(text_parts):
                        if part: # 防止空字符串
                            content_list.append({"type": "text", "text": part})
                        if i < len(new_images):
                            content_list.append({"type": "image", "image": new_images[i]})
                    state['state'].append({
                        "role": "user",
                        "content": content_list
                    })
                elif self.model_type != 'minicpm-v':
                    state['state'].append(
                        {
                            "role": "user",
                            "content": [
                                    {
                                        "type": "image",
                                    }
                                    for _ in range(len(new_images))
                                ]
                                +
                                [
                                    {"type": "text", "text": text},
                                ],
                        })
                    state['images'] += new_images
                else:
                    state['state'].append(
                        {
                            "role": "user",
                            "content": new_images + [text]
                        })
                
            else:
                state['state'].append({"role": "user", "content": text})
            return state
        else:
            raise NotImplementedError(f"不支持{self.task}的任务类型")
    
    def _new_state(self):
        if self.task == 'LLM':
            return []
        elif self.task == 'LMM':
            return {
                "state": [], 
                "images": []
            }
        else:
            raise NotImplementedError(f"不支持{self.task}的任务类型")

    def generate(self, new_text, images=None, state=None,  **generate_kwargs):
        if state is None:
            state = self._new_state()
        
        state = self._add_text(state, new_text, new_images=images)
        if self.task == 'LLM':
            response_message = self.model(state, **generate_kwargs)[0]["generated_text"][-1]
            state.append(response_message)
            return state, response_message['content']
        elif self.task == 'LMM':
            if self.model_type == "qwen3": # qwen3已测试
                inputs = self.processor.apply_chat_template(
                    state['state'],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                # Inference: Generation of the output
                generated_ids = self.model.generate(**inputs, **generate_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )                
                state['state'].append({
                    'role': 'assistant', 
                    'content': output_text[0],
                })
                return state, output_text[0]            
            elif self.model_type != 'minicpm-v':
                text_prompt = self.processor.apply_chat_template(state['state'], add_generation_prompt=True)
                # print(text_prompt)
                if self.device is None:  # 72B这里写的也是CUDA……有可能是device_map=auto可以这么弄
                    inputs = self.processor(
                        text=[text_prompt], images=state['images'], padding=True, return_tensors="pt"
                    ).to('cuda')
                else:
                    inputs = self.processor(
                        text=[text_prompt], images=state['images'], padding=True, return_tensors="pt"
                    ).to(self.device)

                output_ids = self.model.generate(**inputs, **generate_kwargs)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                state['state'].append({
                    'role': 'assistant', 
                    'content': output_text[0],
                })
                return state, output_text[0]
            else:
                answer = self.model.chat(
                    image=None,
                    msgs=state['state'], 
                    tokenizer=self.tokenizer
                )
                state['state'].append({
                    'role': 'assistant', 
                    'content': [answer]
                })
                # print(answer)
                return state, answer

        else:
            raise NotImplementedError(f"不支持{self.task}的任务类型")
    
    def single_qa(self, inputs, state=None, **kwargs):

        images=inputs.get("images",None)
        qs=inputs.get("text",None)
        if len(images) == 1:
            ref_prompt = "# **Instructions for Answering the Question**\n<image>\nAnswer the question about the image.\n\nPlease respond with the letter of the correct option only.\n\n# **Question & Options**\n"
            
        elif len(images) == 2:
            ref_prompt = "# **Instructions for Answering the Question**\nThe first image <image> is the query sample, and the second image <image> is the normal reference sample. Answer the question about the first image.\n\nPlease respond with the letter of the correct option only.\n\n# **Question & Options**\n"
        else:
            raise NotImplementedError("当前仅支持单图或双图的情况，其他情况请自行设计Prompt引导。")
        
        input_text = ref_prompt + qs
        # print(input_text)
        generate_kwargs = {
            "temperature": inputs.get("temperature",0.2),
            "top_p":  inputs.get("top_p",0.7),
            "max_new_tokens": inputs.get("max_new_tokens",1024),
        }

        return self.generate(input_text, state=state, images=images, **generate_kwargs)
    
    def multi_qa(self,inputs,convs,forget_state=False): # maintain train data formation
        state=self._new_state()
        for conv_idx in range(0,len(convs),2):
            cur_qs=convs[conv_idx]["value"].strip()
            inputs["text"]=cur_qs
            if forget_state:
                state=self._new_state()
            state,out=self.single_qa(inputs, state, is_continuation=(conv_idx!=0))
            convs[conv_idx+1]["old_value"]=convs[conv_idx+1]["value"]
            convs[conv_idx+1]["value"]=out
        return convs

class APIModelWrapper:
    """
    Modified to use OpenAI-compatible API instead of local transformers inference.
    Default model set to THUDM/GLM-4.1V-9B-Thinking.
    """
    def __init__(self, model_path='THUDM/GLM-4.1V-9B-Thinking', task='LMM', **kwargs):
        self.task = task
        self.model_path = model_path
        assert task in ['LLM', 'LMM'], f"TransformersModelWrapper不支持{task}类型."
        
        # 获取 API Key 和 Base URL，通常从 kwargs 中获取，也可以硬编码
        api_key = kwargs.get('api_key', "EMPTY") 
        base_url = kwargs.get('base_url', "https://api.siliconflow.cn/v1") # 示例地址，需根据实际服务商修改
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _pil_to_base64(self, image):
        """将 PIL 图片转换为 Base64 字符串"""
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _build_api_messages(self, state):
        """
        将内部状态 (state['state'] + state['images']) 转换为 API 消息格式
        """
        if self.task == 'LLM':
            return state
        
        messages = []
        all_images = state.get('images', [])
        image_idx = 0
        
        for turn in state['state']:
            role = turn['role']
            content = turn['content']
            
            new_content = []
            if isinstance(content, str):
                new_content = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            new_content.append({"type": "text", "text": item['text']})
                        elif item.get('type') == 'image':
                            if image_idx < len(all_images):
                                base64_img = self._pil_to_base64(all_images[image_idx])
                                new_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                                })
                                image_idx += 1
            
            messages.append({"role": role, "content": new_content})
            
        return messages

    def _add_text(self, state, text, new_images=None):
        if self.task == 'LLM':
            assert new_images is None, "语言模型不支持输入图像."
            state.append({"role": "user", "content": text})
            return state
        elif self.task == 'LMM':
            if new_images is not None:
                assert isinstance(new_images, list), "输入的图像应该是一个队列List."
                # 保持原有的内部状态结构
                text_parts = text.split("<image>")
                content_list = []
                for i, part in enumerate(text_parts):
                    if part: # 防止空字符串
                        content_list.append({"type": "text", "text": part})
                    if i < len(new_images):
                        content_list.append({"type": "image",})
                state['state'].append({
                    "role": "user",
                    "content": content_list
                })
                state['images'] += new_images
            else:
                state['state'].append({
                    'role': 'user', 
                    'content': [
                        {
                            'type': 'text', 
                            'text': text
                        }
                    ]  # <--- 封装成列表 + 字典的标准格式
                }) 
            return state
        else:
            raise NotImplementedError(f"不支持{self.task}的任务类型")
    
    def _new_state(self):
        if self.task == 'LLM':
            return []
        elif self.task == 'LMM':
            return {
                "state": [], 
                "images": []
            }
        else:
            raise NotImplementedError(f"不支持{self.task}的任务类型")

    def generate(self, new_text, images=None, state=None, **generate_kwargs):
        if state is None:
            state = self._new_state()
        
        # 更新状态
        state = self._add_text(state, new_text, new_images=images)
        
        # 准备 API 参数
        messages = self._build_api_messages(state)
        
        # 转换生成参数 (映射 transformers 参数到 openai 参数)
        api_kwargs = {
            "model": self.model_path,
            "messages": messages,
            "temperature": generate_kwargs.get("temperature", 0.2),
            "max_tokens": generate_kwargs.get("max_new_tokens", 512),
            "top_p": generate_kwargs.get("top_p", 0.7),
        }

        max_attempts = 5
        output_text = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.chat.completions.create(**api_kwargs)
                output_text = response.choices[0].message.content
                break
            except Exception as e:
                if attempt < max_attempts:
                    print(f"[Warning] API call failed (attempt {attempt}/{max_attempts}): {e}. Retry after 1s...")
                    time.sleep(1)
                else:
                    print(f"[Error] API call failed {max_attempts} times consecutively. Exiting program.")
                    sys.exit(1)

        # 将结果写回状态，保持一致性
        if self.task == 'LLM':
            state.append({"role": "assistant", "content": output_text})
        elif self.task == 'LMM':
            state['state'].append({
                'role': 'assistant', 
                'content': output_text,
            })
            
        return state, output_text
    
    def single_qa(self, inputs, state=None):
        qs = inputs.get("text", None)
        images = inputs.get("images", None)
        if len(images) == 1:
            ref_prompt = "# **Instructions for Answering the Question**\n<image>\nAnswer the question about the image.\n\nPlease respond with the letter of the correct option only.\n\n# **Question & Options**\n" # Please answer the question with the rationale
            
        elif len(images) == 2:
            ref_prompt = "# **Instructions for Answering the Question**\nThe first image <image> is the query sample, and the second image <image> is the normal reference sample. Answer the question about the first image.\n\nPlease respond with the letter of the correct option only.\n\n# **Question & Options**\n"
        else:
            raise NotImplementedError("当前仅支持单图或双图的情况，其他情况请自行设计Prompt引导。")
        
        input_text = ref_prompt + qs

        generate_kwargs = {
            "temperature": inputs.get("temperature", 0.2),
            "top_p":  inputs.get("top_p", 0.7),
            "max_new_tokens": inputs.get("max_new_tokens", 512),
        }

        return self.generate(input_text, state=state, images=images, **generate_kwargs)
    
    def multi_qa(self, inputs, convs, forget_state=False): # maintain train data formation
        state = self._new_state()
        for conv_idx in range(0, len(convs), 2):
            cur_qs = convs[conv_idx]["value"].strip()
            inputs["text"] = cur_qs
            inputs["task_type"] = inputs["task_types"][conv_idx // 2]
            if forget_state:
                state = self._new_state()
            
            state, out = self.single_qa(inputs, state)
            
            convs[conv_idx+1]["old_value"] = convs[conv_idx+1]["value"]
            convs[conv_idx+1]["value"] = out
        return convs

class AGENT_API:
    """
    基于Qwen3VL的LMM AGENT框架
    """
    def __init__(self, model_path, task='LMM', **kwargs):
        
        self.one_shot = kwargs.get("one_shot", False) # 是否开启 one_shot
        self.ds_cls = kwargs.get("ds_cls", "MVTec") # 数据集名称

        # 初始化所有专家实例
        self.ref_extractor = ReferenceExtractor()
        self.knowledge_expert = KnowledgeExpert(one_shot=self.one_shot, ds_cls=self.ds_cls)
        self.logic_expert = LogicExpert(one_shot=self.one_shot, ds_cls=self.ds_cls)
        self.reason_expert = ReasonExpert(one_shot=self.one_shot, ds_cls=self.ds_cls)
        self.decision_maker = DecisionMaker(one_shot=self.one_shot, ds_cls=self.ds_cls)

        # 定义任务路由表 后续根据 self.one_shot与ad_cls选择性补充
        self.task_routing = {
            "Anomaly Detection": ["decision"],
            "Defect Classification": ["know", "decision"],
            "Defect Localization": ["decision"],
            "Defect Description": ["know", "reason", "decision"],
            "Defect Analysis": ["know", "reason", "decision"],
            
            "Object Classification": ["know", "decision"],
            "Object Structure": ["decision"],
            "Object Details": ["decision"],
            "Object Analysis": ["decision"],
        }
        
        self.task = task
        self.model_path = model_path
        assert task in ['LLM', 'LMM'], f"TransformersModelWrapper不支持{task}类型."
        
        # 获取 API Key 和 Base URL，通常从 kwargs 中获取，也可以硬编码
        api_key = kwargs.get('api_key', "EMPTY") 
        base_url = kwargs.get('base_url', "https://api.siliconflow.cn/v1") # 示例地址，需根据实际服务商修改
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _pil_to_base64(self, image):
        """将 PIL 图片转换为 Base64 字符串"""
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _build_api_messages(self, state):
        """
        将内部状态 (state['state'] + state['images']) 转换为 API 消息格式
        """
        if self.task == 'LLM':
            return state
        
        messages = []
        all_images = state.get('images', [])
        image_idx = 0
        
        for turn in state['state']:
            role = turn['role']
            content = turn['content']
            
            new_content = []
            if isinstance(content, str):
                new_content = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            new_content.append({"type": "text", "text": item['text']})
                        elif item.get('type') == 'image':
                            if image_idx < len(all_images):
                                base64_img = self._pil_to_base64(all_images[image_idx])
                                new_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                                })
                                image_idx += 1
            
            messages.append({"role": role, "content": new_content})
            
        return messages

    def _plan_task(self, task_type: str, ad_cls: str) -> str:
        """
        Decision Planning: 根据任务类型与异常类型，分配专家模块
        """
        # return ["decision"] # 屏蔽专家模块 直接决策模块输出结果
        origin_experts = self.task_routing.get(task_type, ["decision"])
        active_experts = origin_experts.copy()
        if ad_cls == "logical":
            active_experts.insert(0, "mask2text")#!!!!!!!!!需要修改
        
        return active_experts # 默认 fallback

    def _add_text(self, state, text, new_images=None):
        if self.task == 'LLM':
            assert new_images is None, "语言模型不支持输入图像."
            state.append({"role": "user", "content": text})
            return state
        elif self.task == 'LMM':
            if new_images is not None:
                assert isinstance(new_images, list), "输入的图像应该是一个队列List."
                text_parts = text.split("<image>")
                content_list = []
                for i, part in enumerate(text_parts):
                    if part: # 防止空字符串
                        content_list.append({"type": "text", "text": part})
                    if i < len(new_images):
                        content_list.append({"type": "image",})
                state['state'].append({
                    "role": "user",
                    "content": content_list
                })
                state['images'] += new_images
            else:
                state['state'].append({
                    'role': 'user', 
                    'content': [
                        {
                            'type': 'text', 
                            'text': text
                        }
                    ]  # <--- 封装成列表 + 字典的标准格式
                }) 
            return state
        else:
            raise NotImplementedError(f"不支持{self.task}的任务类型")
    
    def _new_state(self):
        if self.task == 'LLM':
            return []
        elif self.task == 'LMM':
            return {
                "state": [], 
                "images": []
            }
        else:
            raise NotImplementedError(f"不支持{self.task}的任务类型")

    def generate(self, new_text, images=None, state=None, **generate_kwargs):
        if state is None:
            state = self._new_state()
        # 更新状态
        state = self._add_text(state, new_text, new_images=images)
        # 准备 API 参数
        messages = self._build_api_messages(state)
        # 转换生成参数 (映射 transformers 参数到 openai 参数)
        api_kwargs = {
            "model": self.model_path,
            "messages": messages,
            "temperature": generate_kwargs.get("temperature", 0.2),
            "max_tokens": generate_kwargs.get("max_new_tokens", 512),
            "top_p": generate_kwargs.get("top_p", 0.7),
        }
        
        max_attempts = 5
        output_text = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.chat.completions.create(**api_kwargs)
                output_text = response.choices[0].message.content
                break
            except Exception as e:
                if attempt < max_attempts:
                    print(f"[Warning] API call failed (attempt {attempt}/{max_attempts}): {e}. Retry after 1s...")
                    time.sleep(1)
                else:
                    print(f"[Error] API call failed {max_attempts} times consecutively. Exiting program.")
                    sys.exit(1)

        # 将结果写回状态，保持一致性
        if self.task == 'LLM':
            state.append({"role": "assistant", "content": output_text})
        elif self.task == 'LMM':
            state['state'].append({
                'role': 'assistant', 
                'content': output_text,
            })
            
        return state, output_text

    def process(self, images=None, state=None, task_info:dict={}, **generate_kwargs): 
        if state is None:
            state = self._new_state()
        
        query = task_info.get("query", None)
        task_type = task_info.get("task_type", "Anomaly Detection")
        ad_cls = task_info.get("ad_cls", "structural")
        
        # 1. 规划 (Planning)
        active_experts = self._plan_task(task_type, ad_cls)   # 根据关键词确定任务类别
        
        if self.one_shot:
            active_experts.insert(0, "ref")
        
        context_buffer = {} # 根据任务类型构建一个context_buffer
        context_buffer["task_routing"] = active_experts
        # 2. 选择性执行专家模块 (Execution Pipeline)
        if "ref" in active_experts:
            # 此处为伪代码 将查询得到的图像进行处理后 返回PLI图像列表
            context_buffer["ReferenceExtractor"] = self.ref_extractor.execute(task_info)
        if "mask2text" in active_experts:
            # 此处为伪代码 将逻辑异常包含的掩码转换成固定格式的逻辑描述 返回相关提示词
            context_buffer["LogicExpert"] = self.logic_expert.execute(task_info)
        if "know" in active_experts:    # [Expert: Knowledge Expert]
            context_buffer["KnowledgeExpert"] = self.knowledge_expert.execute(task_info)
        if "reason" in active_experts:  # [Expert: Reason Expert]
            context_buffer["ReasonExpert"] = self.reason_expert.execute(task_info)
        
        sys_prompt = self.decision_maker.execute(task_info, context_buffer)
        user_prompt = f"# **Instructions for Answering the Question**\n{sys_prompt}\n\n# **Question & Options**\n{query}" # 把问题放到引导的最后
        # print(user_prompt)
        return self.generate(new_text=user_prompt, images=images, state=state, **generate_kwargs)  # 根据初始的信息输出生成理据
    
    def single_qa(self, inputs, state=None):# 增加了ROI图像区域分割引导
        images_list = inputs.get("images_list", None)
        ad_cls = inputs.get("ad_class", "structural")
        generate_kwargs = {
            "temperature": inputs.get("temperature",0.2),
            "top_p":  inputs.get("top_p",0.7),
            "max_new_tokens": inputs.get("max_new_tokens",512),
        }

        task_info = {
            "query": inputs.get("text",None),
            "img_num": len(images_list) if images_list is not None else 0,
            "task_type": inputs.get("task_type", None),
            "image_path": inputs.get("image_path", None),
            "seg_path": inputs.get("seg_path", None),
            "ad_cls": ad_cls,
            "obj_cls": inputs.get("obj_class", None),
        }
        return self.process(
            images=images_list, 
            state=state, 
            task_info=task_info,
            **generate_kwargs)
    
    def multi_qa(self,inputs,convs,forget_state=False): # maintain train data formation
        state=self._new_state()
        for conv_idx in range(0,len(convs),2):
            cur_qs=convs[conv_idx]["value"].strip()
            inputs["text"]=cur_qs
            inputs["task_type"] = inputs["task_types"][conv_idx // 2]
            
            if forget_state:
                state=self._new_state()
            state,out=self.single_qa(inputs,state)
            convs[conv_idx+1]["old_value"]=convs[conv_idx+1]["value"]
            convs[conv_idx+1]["value"]=out
        return convs