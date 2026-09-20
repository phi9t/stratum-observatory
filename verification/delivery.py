"""Read delivered entry points from the application registry."""
import json, subprocess
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=None)
def page_inventory(root):
    registry=Path(root).resolve()/'src/pages.mjs'
    return json.loads(subprocess.check_output(['node','--input-type=module','-e',
        'const {PAGES}=await import(process.argv[1]);console.log(JSON.stringify(PAGES));',
        registry.as_uri()],text=True))

def page_for_model(root,model,view=None):
    shared=next((p for p in page_inventory(root) if p.get('view') and p['view']==view),None)
    if shared:return shared
    return next((p for p in page_inventory(root) if model in p['models']),None)

def artifact_name(root,model,view=None):
    page=page_for_model(root,model,view)
    return page['file'] if page else 'index.html'

def artifact_names(root):
    return {'index.html',*(p['file'] for p in page_inventory(root))}
