import argparse
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from lmm_model import TransformersModelWrapper, CoExpert, APIModelWrapper, AGENT_API, prepare_inputs_with_masks
from lmm_ulits import *

'''=============================== Global Config ==================================='''
API_PATH = "" # model API path
API_KEY = "" # your API key for model API inference, if needed.
BASE_URL = "" # base url for API, if needed.

MODEL_NAME = "Qwen3-VL-8B-Instruct"
MODEL_TYPE = "CoExpert"# ["CoExpert", "Transformers", "API", "AGENT_API"]
MODEL_PATH = "../Checkpoints/Qwen3-VL-8b-Instruct"
MODEL_ADAPTER_PATH = None # Optional LoRA adapter path; keep None when not used.

DATASET_NAME = "all"
DATASET_FROM = "mmad_eval.json"
DATASET_TO = "answer_mmad.json"
DATASET_START = 0

REF_ONE_SHOT = True
REF_IMG_TYPE = "similar"  # "random", "similar"
REF_IMG_NUM = 1
'''================================================= Evaluation ================================================='''

def main():
    # ------------------------------------ 1. Parse Arguments ------------------------------------
    parser = argparse.ArgumentParser()
    # api config
    parser.add_argument("--api_key", type=str, default=API_KEY)
    parser.add_argument("--api_path", type=str, default=API_PATH)
    parser.add_argument("--base_url", type=str, default=BASE_URL)
    
    # model config
    parser.add_argument("--model_name", type=str, default=MODEL_NAME) # ["llava-onevision-qwen2-7b-ov", "GLM-4.1V-9B-Thinking", "xxx"]
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE) # ["LLaVA-NeXT", "transformers", "transformers-api"]
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--adapter_path", type=str, default=MODEL_ADAPTER_PATH)

    # dataset config (must be configured)
    parser.add_argument("--ds", type=str, default=DATASET_NAME) # mvtec, mvtec-loco, goodsad, visa
    parser.add_argument("--data_path", type=str, default="data", help="DO NOT add / behind data_path if you want directly the dataset dir")
    parser.add_argument("--subset", type=str, default="all") # name of subsets in eval data, separate with ',' if multiple subsets
    parser.add_argument("--from", type=str, default=DATASET_FROM, dest='question_file')
    parser.add_argument("--to", type=str, default=DATASET_TO, dest='answer_file')
    parser.add_argument("--start", type=int, default=DATASET_START) # start from mid, for resuming eval

    # evaluation config
    parser.add_argument("--one_shot", type=bool, default=REF_ONE_SHOT) # True False
    parser.add_argument("--ref_type", type=str, default=REF_IMG_TYPE) # "random", "similar"
    parser.add_argument("--ref_num", type=int, default=REF_IMG_NUM) # number of reference images to use
    parser.add_argument("--mask_mode", type=str, default='default', choices=['all_none', 'all_empty', 'empty_when_missing', 'default']) # default: read from eval data, none when missing
    parser.add_argument("--ref_mode",  type=str, default='default', choices=['all_empty', 'copy_query', 'default']) # default: read from eval data
    parser.add_argument("--ref_mask_mode",  type=str, default='default', choices=['none_for_ref', 'empty_for_ref', 'copy_query_mask', 'default', 'follow_mask_mode']) # default: use mask from eval data, if none copy query mask 
    parser.add_argument("--qa_mode",  type=str, default='single_qa', choices=['single_qa', 'default']) # default: multi_qa according to eval data
    parser.add_argument("--image_num", type=int, default=None) # manaually set image num, will auto pad <image> and empty picture
    parser.add_argument("--overwrite_image_aspect_ratio", type=str, default="randomroi")
    parser.add_argument("--use_pad_when_db", action='store_true') # force image_aspect_ratio use pad when input images > 1
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()
    
    # ------------------------------------ 2. Load Evaluation Data ------------------------------------
    ds_name=args.ds
    data_dir=args.data_path
    start=args.start
    if data_dir.endswith("/") and not data_dir.endswith(ds_name+"/"): # guess it is on ./evaluate/
        data_dir=os.path.join(data_dir,ds_name)
    question_path=os.path.join(data_dir,args.question_file)
    answer_path=os.path.join(data_dir,args.answer_file)
    
    all_datas=load_mmad_data(question_path, args.one_shot, args.ref_type, args.ref_num)
    
    if args.subset!="all":
        subsets=args.subset.split(",")
    else:
        subsets=None
    
    # ------------------------------------ 3. Initialize Model ------------------------------------
    wrapper=None    
    if args.model_type == "CoExpert":   # Local inference via CoExpert framework
        print(f"Initializing local model with CoExpert framework, base model: {args.model_name}...")
        wrapper=CoExpert(
                model_path=args.model_path,
                task='LMM',
                device='cuda',
                ds_cls=ds_name,
                one_shot=args.one_shot,
                adapter_path=args.adapter_path,
            ) 
    elif args.model_type == "Transformers":   # Local inference via transformers
        print(f"Initializing local model with transformers, model: {args.model_name}...")
        wrapper=TransformersModelWrapper(
                model_path=args.model_path,
                task='LMM',
                ds_cls=ds_name,
                device='cuda',
            )
    elif args.model_type == "API":   # Model API inference
        print(f"Initializing API client, model: {args.model_name}...")
        wrapper=APIModelWrapper(
                model_path=args.api_path,
                task='LMM',
                ds_cls=ds_name,
                one_shot=args.one_shot,
                api_key=args.api_key,
                base_url=args.base_url,
            )    
    elif args.model_type == "AGENT_API":   # Model API inference
        print(f"Initializing AGENT_API client, model: {args.model_name}...")
        wrapper=AGENT_API(
                model_path=args.api_path,
                task='LMM',
                ds_cls=ds_name,
                one_shot=args.one_shot,
                api_key=args.api_key,
                base_url=args.base_url,
            ) 
    else:
        raise ValueError(f"unkown model_type: {args.model_type}")

    # ------------------------------------ 4. Evaluation Loop ------------------------------------
    all_ans=[]
    # main cycle
    for d in tqdm(all_datas[start:]):
        
        subset=d["subset"]
        if subsets is not None and subset not in subsets:
            continue
        
        # image process
        if "src" in d:
            img_p=d["src"]
        elif "image" in d and d["image"] is not None:
            img_p=d["image"]
            if type(img_p) is str:
                img_p=[img_p]
        else:
            img_p=[]
        
        imgs = [load_and_resize(os.path.join(data_dir, i), mode='RGB') for i in img_p]
        
        # image num correct
        if args.image_num is None:
            img_num=len(imgs)
        else:
            img_num=args.image_num
            
        if img_num<len(imgs):
            print(f"WARNING: image_num {img_num} is smaller than len(images) {len(imgs)}, will chunk")
            imgs=imgs[:img_num]
        elif img_num>len(imgs):
            print(f"WARNING: image_num {img_num} is larger than len(images) {len(imgs)}, will pad")
            img_e=Image.new("RGB",imgs[0].size,(0,0,0))
            imgs=imgs+[img_e]*(img_num-len(imgs))
        
        assert img_num==len(imgs)
            
        # ref process
        img_q=imgs[0]
        img_r=imgs[1:]
        if args.ref_mode=="all_empty":
            img_r=[Image.new("RGB",i.size,(0,0,0)) for i in img_r]
        elif args.ref_mode=="copy_query":
            img_r=[img_q]*(img_num-1)
        elif args.ref_mode=="default":
            pass
        else:
            raise ValueError(f"unkown ref_mode {args.ref_mode}")
        imgs=[img_q]+img_r
        
        assert img_num==len(imgs), "after ref process"
            
        # mask process
        if "mask" in d and d["mask"] is not None:
            mask_p=d["mask"]
            if type(mask_p) is str:
                mask_p=[mask_p]
            masks = [load_and_resize(os.path.join(data_dir,i), mode='L') for i in mask_p]
        else:
            masks=None
        # mask num correct
        mask_o=masks
        if masks is None:
            mask_o=[]
        mask_numr=len(imgs)-len(mask_o)
        if mask_numr>0:
            mask_o=mask_o+[None]*(mask_numr)
        elif mask_numr<0:
            mask_o=mask_o[:mask_numr]
            
        # mask control
        if args.mask_mode=="all_none":
            masks=None
        elif args.mask_mode=="all_empty":
            masks=[]
            for img in imgs:
                img_np=img.numpy()
                masks.append(np.zeros(img_np.shape[:2],dtype=img_np.dtype))
        elif args.mask_mode=='empty_when_missing':
            masks=[]
            for idx,img in enumerate(imgs):
                if mask_o[idx] is None:
                    img_np=img.numpy()
                    masks.append(np.zeros(img_np.shape[:2],dtype=img_np.dtype))
                else:
                    masks.append(mask_o[idx])
        elif args.mask_mode=='default':
            masks=mask_o # no process, none when missing
        else:
            raise ValueError(f"unkown mask_mode {args.mask_mode}")
        
        # ref mask control
        mask_q=masks[0]
        mask_r=masks[1:]
        mask_rnum=len(mask_r)
        if args.ref_mask_mode=="none_for_ref":
            mask_r=[None]*mask_rnum
        elif args.ref_mask_mode=='empty_for_ref':
            mask_r=[]
            for img in imgs[1:]:
                img_np=img.numpy()
                mask_r.append(np.zeros(img_np.shape[:2],dtype=img_np.dtype))
        elif args.ref_mask_mode=='copy_query_mask':
            mask_r=[mask_q]*mask_rnum
        elif args.ref_mask_mode== 'default': # use eval_data, copy query if missing
            mask_r=[]
            for idx,img in enumerate(imgs[1:],start=1):
                if mask_o[idx] is None:
                    mask_r.append(mask_q)
                else:
                    mask_r.append(mask_o[idx])
        elif args.ref_mask_mode== 'follow_mask_mode':
            pass # no process
        else:
            raise ValueError(f"unkown ref_mask_mode {args.ref_mask_mode}")
        masks=[mask_q]+mask_r

        # bbox process
        if "bbox" in d and d["bbox"] is not None: # no process
            boxes_list=d["bbox"]
        else:
            boxes_list=None
        
        assert wrapper is not None
        ans=deepcopy(d)        
        # default inputs params
        if args.one_shot:
            img_ref=[load_and_resize(os.path.join(data_dir, i), mode='RGB') for i in ans.get("ref_image", None)]
            imgs.extend(img_ref)
        inputs={
                "temperature": 0.2,
                "top_p": 0.7,
                "max_new_tokens": args.max_new_tokens,
                "use_pad_when_db": args.use_pad_when_db, 
                "images": imgs,
                "masks": masks,
                "boxes_list": boxes_list,
        }
        # Additional fields
        inputs["origin_path"] = ans.get("origin_path", None)
        inputs["image_path"] = ans.get("image", None)
        inputs["seg_path"] = ans.get("seg_masks", None)
        inputs["obj_class"] = ans.get("subset", None)
        inputs["task_types"] = ans.get("types", None)
        inputs["size_bd"] = ans.get("size_bd", None)
        if inputs["task_types"] is not None:
            inputs["task_type"] = inputs["task_types"][0]
        else:
            inputs["task_type"] = None
        inputs["ad_class"] = ans.get("ad_class", None)
        
        if args.model_type == "CoExpert" or args.model_type == "AGENT_API":
            inputs = prepare_inputs_with_masks(inputs, args.one_shot)
        
        # qa control
        qs=d["conversations"]
        if type(qs) is list: # assert qs is convs
            if args.qa_mode=="single_qa":
                convs=wrapper.multi_qa(inputs,qs,forget_state=True)
            elif args.qa_mode=="default": #default
                convs=wrapper.multi_qa(inputs,qs)
            else:
                raise ValueError(f"unkown qa_mode {args.qa_mode}") 
            ans["conversations"]=convs
        elif type(qs) is str:
            _,out=wrapper.single_qa(inputs)
            ans["conversations"]=make_convs([qs,out])
        else:
            raise ValueError(f"unkown qs type {type(qs)}")
        
        # write ans
        data_write(ans,answer_path,"jsonl","a")
        all_ans.append(ans)

    # ------------------------------------ 5. Save Final Outputs ------------------------------------
    # main cycle finished
    data_write(all_ans,answer_path,"json","w")
    jsonl_tmp=answer_path.replace(".json",".jsonl")
    if os.path.isfile(jsonl_tmp):
        os.remove(jsonl_tmp)

if __name__ == "__main__":
    main()