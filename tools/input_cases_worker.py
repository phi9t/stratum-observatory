"""Offline child process: only pinned formatter, tokenizer and pixel preprocessing."""
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image, ImageDraw
from tokenizers import Tokenizer
import PIL
import tokenizers
import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(sys.argv[1])
    encoding = load_module('encoding', root / 'encoding.py')
    processor = load_module('image_processor', root / 'image_processor.py')
    config = json.loads((root / 'config.json').read_text())
    tokenizer_config = json.loads((root / 'tokenizer_config.json').read_text())
    assert 'chat_template' not in tokenizer_config
    assert not tokenizer_config['add_bos_token'] and not tokenizer_config['add_eos_token']
    tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
    class Adapter:
        unk_token_id = None
        def convert_tokens_to_ids(self, token):
            return tokenizer.token_to_id(token)
        def encode(self, prompt):
            return tokenizer.encode(prompt, add_special_tokens=False).ids
    adapter = Adapter()
    v = config['vision_config']
    args = SimpleNamespace(vision_patch_size=v['patch_size'], vision_downsample_ratio=v['downsample_ratio'],
        vision_max_n_token=v['max_image_tokens'], vision_min_pixels=v['min_pixels'],
        vision_max_wh_ratio=v['max_wh_ratio'], image_token_id=config['image_token_id'],
        vision_enabled=v['num_hidden_layers'] > 0)
    chart = {'width':126,'height':84,'background':'#ffffff','baseline':70,'bars':[
        {'label':'A','x':14,'y':42,'width':21,'height':28,'color':'#286d8a'},
        {'label':'B','x':49,'y':14,'width':21,'height':56,'color':'#986323'},
        {'label':'C','x':84,'y':28,'width':21,'height':42,'color':'#6c5694'}]}
    image = Image.new('RGB', (chart['width'], chart['height']), chart['background'])
    draw = ImageDraw.Draw(image)
    for bar in chart['bars']:
        draw.rectangle((bar['x'],bar['y'],bar['x']+bar['width']-1,bar['y']+bar['height']-1), fill=bar['color'])
        draw.text((bar['x']+7,chart['baseline']+2),bar['label'],fill='#000000')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    png = buffer.getvalue()
    chart['png_base64'] = base64.b64encode(png).decode('ascii')
    chart['png_sha256'] = hashlib.sha256(png).hexdigest()
    specs = [
        ('text-chat','Multi-turn text chat',[
            {'role':'system','content':'Answer briefly.'},
            {'role':'user','content':'Name a primary color.'},
            {'role':'assistant','content':'Blue.'},
            {'role':'user','content':'Name another.'}]),
        ('chart-question','A question about an authored chart',[
            {'role':'user','content':[{'type':'image','data':chart['png_base64']},
                {'type':'text','text':'Which bar is tallest?'}]}])]
    cases = []
    names = {-1:'TEXT',0:'IMAGE_START',1:'IMAGE',2:'IMAGE_NEW_LINE',3:'IMAGE_END'}
    for id_,title,messages in specs:
        prompt, media = encoding.encode_messages(messages, thinking_mode='chat', return_multi_modal_data=True)
        serialized = tokenizer.encode(prompt, add_special_tokens=False)
        ids, types, images = processor.prepare_vl_inputs(prompt,media['images'],adapter,args)
        image_record = None
        if images:
            im = images[0]
            gh, gw = processor.llm_grid(im.n_vit_h*args.vision_patch_size, im.n_vit_w*args.vision_patch_size,
                args.vision_patch_size,args.vision_downsample_ratio)
            image_positions = [i for i,t in enumerate(types) if t == processor.IMAGE]
            ratio = args.vision_downsample_ratio
            patches = []
            merged = [{'index':r*gw+c,'row':r,'column':c,'token_index':image_positions[r*gw+c], 'patch_indices':[]}
                for r in range(gh) for c in range(gw)]
            for r in range(im.n_vit_h):
                for c in range(im.n_vit_w):
                    idx = r*im.n_vit_w+c
                    mi = (r//ratio)*gw+c//ratio
                    patches.append({'index':idx,'row':r,'column':c,'merged_index':mi,'token_index':image_positions[mi]})
                    merged[mi]['patch_indices'].append(idx)
            image_record = {'chart':chart,'patch_size':args.vision_patch_size,
                'patch_grid':[im.n_vit_h,im.n_vit_w],'merged_grid':[gh,gw],
                'span':[im.start,im.start+len(im.types)],'patches':patches,'merged':merged,
                'preprocessing_args':vars(args),'publisher_default_grid':list(processor.plan_image_grid(image.width,image.height,args)),
                'patch_tensor_shape':list(im.patches.shape),'patch_tensor_dtype':str(im.patches.dtype),
                'patch_tensor_sha256':hashlib.sha256(im.patches.view(torch.uint8).numpy().tobytes()).hexdigest(),
                'scope':'Exact publisher preprocessing and image span; patch-to-merge ownership is a spatial layout, not executed vision weights.'}
        displayed_messages = json.loads(json.dumps(messages))
        if images:
            displayed_messages[0]['content'][0] = {'type':'image','asset':'authored-chart.png'}
        cases.append({'id':id_,'title':title,'messages':displayed_messages,'thinking_mode':'chat',
            'serialized_prompt':prompt,'serialized_token_ids':serialized.ids,'serialized_token_pieces':serialized.tokens,
            'token_ids':ids,'token_pieces':[tokenizer.id_to_token(i) for i in ids],'token_types':types,
            'positions':[{'index':i,'token_id':token,'piece':tokenizer.id_to_token(token),'type':typ,'type_name':names[typ]}
                for i,(token,typ) in enumerate(zip(ids,types))], 'image':image_record})
    print(json.dumps({'cases':cases,'execution':{'formatter':'encoding.encode_messages','tokenizer':'tokenizers.Tokenizer.encode(add_special_tokens=False)',
        'image_processor':'image_processor.prepare_vl_inputs','chat_template_present':False,
        'status':'EXECUTED_CPU','weights_loaded':False,'network':'denied','vision_weights':'NOT_RUN',
        'versions':{'tokenizers':tokenizers.__version__,'pillow':PIL.__version__,'torch':torch.__version__}}},ensure_ascii=False))


if __name__ == '__main__':
    main()
