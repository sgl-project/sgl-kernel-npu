from ._move_rows import move_rows


def move_mla_kv_cache(
    k_cache, rope_cache, tgt_loc, src_loc, page_size, use_fia_nz=False
):
    move_rows(k_cache, rope_cache, tgt_loc, src_loc, page_size, use_fia_nz)
