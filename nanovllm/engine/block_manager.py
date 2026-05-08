"""
KV Cache 块管理器（BlockManager）。

核心思想来自 PagedAttention 论文：把 KV Cache 按固定大小的 block 管理，
类似操作系统的虚拟内存分页。好处：
1. 避免为每个序列预分配最大长度的连续内存（减少浪费）
2. 支持 Prefix Caching：相同前缀的序列可以共享 KV Cache block
3. 支持 Preemption：可以释放某个序列的 block 给其他序列使用

Block 的生命周期：
  free → allocated（被某个序列使用）→ deallocated → free
  deallocated 后如果 hash 还在，可以被后续请求作为 prefix cache 复用
"""

from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """一个物理 KV Cache block 的元数据"""

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0      # 引用计数：有多少个序列在使用这个 block
        self.hash = -1          # 内容 hash（用于 prefix caching 匹配）
        self.token_ids = []     # 这个 block 存储的 token 内容（用于验证 hash 碰撞）

    def update(self, hash: int, token_ids: list[int]):
        """block 被填满后，记录其 hash 和内容（供后续 prefix cache 查找）"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """block 被重新分配时，清空元数据"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    管理所有物理 KV Cache block 的分配、释放和 prefix cache 查找。

    数据结构：
    - blocks: 所有物理 block 的元数据数组
    - hash_to_block_id: hash → block_id 的映射（prefix cache 索引）
    - free_block_ids: 空闲 block 队列
    - used_block_ids: 正在使用的 block 集合
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算一个 block 的 hash。使用链式 hash：
        hash(block_i) = hash(block_i 的 token_ids + hash(block_{i-1}))
        这样相同前缀的不同序列会得到相同的 hash 链。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """从空闲队列分配一个 block"""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 如果这个 block 之前有 hash 记录，清除它（因为内容要被覆盖了）
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """释放一个 block 回空闲队列（注意：hash 信息保留，可供 prefix cache 复用）"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """
        检查是否有足够的 block 来容纳这个序列。
        同时计算 prefix cache 命中了多少个 block。

        返回值：
        - >= 0: 可以分配，返回命中的 cached block 数
        - -1: 空间不足，无法分配
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        # 逐 block 检查是否有 prefix cache 命中（最后一个 block 不检查，因为可能没填满）
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 验证 hash 没有碰撞（内容必须完全匹配）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 如果 cached block 还在 used 中，不需要额外分配
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """
        为序列分配 block table。
        - 前 num_cached_blocks 个 block 复用已有的（prefix cache 命中）
        - 剩余的从空闲队列分配新 block
        """
        assert not seq.block_table
        h = -1
        # 复用 cached blocks
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                # block 在 free 队列中但 hash 还在，重新激活它
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 分配新 blocks
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        # 标记已缓存的 token 数（这些 token 不需要重新计算 KV）
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有 block（引用计数减 1，归零时真正释放）"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        decode 时检查是否能追加一个 token。
        只有当 token 恰好是新 block 的第一个时，才需要分配新 block。
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """如果当前 token 是新 block 的第一个，分配一个新 block"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """
        对刚刚完成计算的 block 计算 hash 并注册到 prefix cache 索引。
        只对完整填满的 block 计算 hash（不完整的 block 不能被复用）。
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
