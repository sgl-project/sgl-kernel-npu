def move_rows(a, b, tgt_loc, src_loc, page_size=None, nz=False):
    src, dst = src_loc.reshape(-1).cpu().tolist(), tgt_loc.reshape(-1).cpu().tolist()
    if len(src) != len(dst):
        raise ValueError("source and target location counts must match")

    def row(cache, loc):
        page, slot = divmod(loc, page_size or cache.shape[2])
        if nz:
            return cache.view(cache.shape[0], cache.shape[1], -1, page_size, 16)[
                :, page, :, slot, :
            ]
        return cache[:, page, slot]

    snapshots = [(row(a, s).clone(), row(b, s).clone()) for s in src]
    for s, t, (av, bv) in zip(src, dst, snapshots):
        if s != t:
            row(a, t).copy_(av)
            row(b, t).copy_(bv)
