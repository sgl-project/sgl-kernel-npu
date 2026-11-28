from ._move_rows import move_rows


def move_kv_cache_rows(keys, values, tgt_loc, src_loc):
    move_rows(keys, values, tgt_loc, src_loc)
