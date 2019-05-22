from copy import copy

DS_CFG_NO_SW = {
        'splits' : {
            'train' : 80,
            'val' : 10,
            'test' : 10
            },
        'stopword_removal' : True
        }

DS_CFG_SW = copy(DS_CFG_NO_SW)
DS_CFG_SW['stopword_removal'] = False
