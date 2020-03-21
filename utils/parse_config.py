def parse_cfg(cfgfile):
    '''
    takes a configuration file
    returns a list of blocks. Each block describes a block in the neural network to be built.
    block is represented as a dictionary in the list
    '''
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0] # get rid of empty lines
    lines = [x for x in lines if x[0] != '#'] # get rid of comment lines
    lines = [x.lstrip().rstrip() for x in lines] # get rid of frange whitespaces
    
    blocks = []
    block = {}

    for line in lines:
        if line[0] == '[': # start of the new block
            if len(block) != 0: # content of the last block 
                blocks.append(block)
                block = {} # re-init dict for the next block
                block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block) # store the last one

    return blocks
    
