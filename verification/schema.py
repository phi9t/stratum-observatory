"""Small explicit JSON-Schema subset used by this project (not a general validator)."""
import re

# Keywords a schema author might reasonably expect to work. This evaluator does not
# implement them, so it must say so instead of quietly accepting the value: a constraint
# that is silently ignored is worse than one that was never written.
UNSUPPORTED = ('oneOf','anyOf','allOf','not','patternProperties','propertyNames',
               'minimum','maximum','exclusiveMinimum','exclusiveMaximum',
               'uniqueItems','maxItems','maxLength','minProperties','maxProperties',
               'dependentRequired','if','then','else','const','contains')

def validate(value,schema,*,root=None,path='$'):
    root=schema if root is None else root
    if '$ref' in schema:
        ref=schema['$ref']
        if not ref.startswith('#/$defs/'):return [f'{path}: unsupported/nonlocal schema reference']
        return validate(value,root['$defs'][ref.split('/')[-1]],root=root,path=path)
    errors=[]
    def fail(msg):errors.append(f'{path}: {msg}')
    for keyword in UNSUPPORTED:
        if keyword in schema:fail(f'unsupported schema keyword {keyword}')
    if 'enum' in schema and value not in schema['enum']:fail('value not in enum')
    checks={'null':lambda v:v is None,'boolean':lambda v:type(v)==bool,'integer':lambda v:type(v)==int,'number':lambda v:type(v) in (int,float),'string':lambda v:isinstance(v,str),'object':lambda v:isinstance(v,dict),'array':lambda v:isinstance(v,list)}
    types=schema.get('type');types=[types] if isinstance(types,str) else types
    if types and not any(checks[t](value) for t in types):fail('incorrect type');return errors
    if isinstance(value,str):
        if len(value)<schema.get('minLength',0):fail('string too short')
        if 'pattern' in schema and not re.search(schema['pattern'],value):fail('pattern mismatch')
    if isinstance(value,list):
        if len(value)<schema.get('minItems',0):fail('array too short')
        if 'items' in schema:
            for i,x in enumerate(value):errors+=validate(x,schema['items'],root=root,path=f'{path}[{i}]')
    if isinstance(value,dict):
        for key in schema.get('required',[]):
            if key not in value:fail(f'missing {key}')
        props=schema.get('properties',{})
        extra=schema.get('additionalProperties')
        if extra is False:
            for key in value.keys()-props.keys():fail(f'unknown {key}')
        for key,x in value.items():
            if key in props:errors+=validate(x,props[key],root=root,path=f'{path}.{key}')
            elif isinstance(extra,dict):errors+=validate(x,extra,root=root,path=f'{path}.{key}')
    return errors
