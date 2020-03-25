# import visdom 



def check_format(v_type, data, **kwargs):
    if v_type=='image':
        pass
    elif v_type=='images':
        pass
    elif v_type=='heatmap':
        pass
    else:
        pass

def visualize(env, v_type, data, caption):
    visd = visdom.Visdom(env=env)
    check_format(v_type, data)
    if v_type=='image':
        visd.image(data, opts=dict(caption=caption))
        print('ok')
    elif v_type=='images':
        visd.images(data, opts=dict(caption=caption))
        print('done')
    else:
        pass

