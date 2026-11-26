def verify_ngram_tree_greedy(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
):
    candidates, indices, children, siblings = [
        x.cpu().tolist()
        for x in (candidates, retrive_index, retrive_next_token, retrive_next_sibling)
    ]
    targets = target_predict.reshape(-1).cpu().tolist()
    for b in range(len(candidates)):
        last = indices[b][0]
        accept_index[b, 0] = last
        accepted = 0
        child = children[b][0]
        while child != -1 and accepted < accept_index.shape[1] - 1:
            token = targets[last]
            while child != -1 and candidates[b][child] != token:
                child = siblings[b][child]
            if child == -1:
                break
            predicts.reshape(-1)[last] = token
            accepted += 1
            last = indices[b][child]
            accept_index[b, accepted] = last
            child = children[b][child]
        accept_token_num[b] = accepted
        predicts.reshape(-1)[last] = targets[last]
