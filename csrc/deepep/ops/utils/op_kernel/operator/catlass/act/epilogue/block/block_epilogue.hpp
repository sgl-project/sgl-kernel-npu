#ifndef ACT_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
#define ACT_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP

#include "../../../act/act.hpp"

namespace Act::Epilogue::Block {

template <class DispatchPolicy, class... Args>
class BlockEpilogue
{
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

}  // namespace Act::Epilogue::Block

#include "../../../act/epilogue/block/block_epilogue_per_token_dequant.hpp"
#endif  // ACT_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
