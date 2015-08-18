try:
    # faster implementation using bindings to libxml
    from lxml import etree as ET
except ImportError:
    print 'Falling back to default ElementTree implementation'
    from xml.etree import ElementTree as ET

import re
RE_NAME_LONG = re.compile('^(\d+)_Exp(\d+)Cam(\d+)$')
RE_NAME = re.compile('^Exp(\d+)Cam(\d+)$')


def _parse_xml(x):
    frames = cams = exps = arrays = 0
    for n in [e.get('Name') for e in ET.fromstring(x).findall('Arrays/Array[@Name]')]:
        # Print "n" if you want to see the values of the "Name" attribute
        # print n
        
        arrays += 1
        
        # Names found in Oren's flex files
        m = RE_NAME_LONG.match(n)
        if m:
            frames, exps, cams = [max(g) for g in zip(map(int, m.groups()), (frames, exps, cams))]
            continue
        
        # Names found in Mojca's flex files
        m = RE_NAME.match(n)
        if m:
            exps, cams = [max(g) for g in zip(map(int, m.groups()), (exps, cams))]
            frames = arrays / cams
            continue
        
        raise Exception('Unknown flex name pattern')

    return frames, exps, cams

def _r(fp):
    '''
        Read one byte as char and return byte value
    '''
    return ord(fp.read(1))

'''
    type reading utils
'''
def _get_short(fp):
    return _r(fp) + (_r(fp) << 8)

def _get_int(fp):
    return _r(fp) + (_r(fp) << 8) + (_r(fp) << 16) + (_r(fp) << 24)

def get_flex_data(im):
    im = open(im, 'rb')

    _mm = im.read(2)
    _ver = _get_short(im)
    _offs = _get_int(im)
    im.seek(_offs)
    
    _num_tags = _get_short(im)
    xml = None
    
    for _tag_idx in xrange(_num_tags):
        _tag = _get_short(im)
        _tag_type = _get_short(im)
        _tag_len = _get_int(im)
        if _tag_type == 3 and _tag_len == 1:
            _tag_value = _get_short(im)
            _ = _get_short(im)
        else:
            _tag_value = _get_int(im)
        
        if _tag == 65200:
            _saved_offs = im.tell()
            im.seek(_tag_value)
            xml = im.read(_tag_len)
            im.seek(_saved_offs)
    
    im.close()
    
    return _parse_xml(xml)

if __name__ == "__main__":
    print '001003000.flex frames=%d experiments=%d channels=%d' % get_flex_data('001003000.flex')

