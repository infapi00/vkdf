#include "vkdf-buffer.hpp"
#include "vkdf-memory.hpp"
#include "vkdf-util.hpp"

VkdfBuffer
vkdf_create_buffer(VkdfContext *ctx,
                   VkBufferCreateFlags flags,
                   VkDeviceSize size,
                   VkBufferUsageFlags usage,
                   uint32_t mem_props)
{
   VkdfBuffer buffer;

   // Create buffer object
   VkBufferCreateInfo buf_info;
   buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
   buf_info.pNext = NULL;
   buf_info.usage = usage;
   buf_info.size = size;
   buf_info.queueFamilyIndexCount = 0;
   buf_info.pQueueFamilyIndices = NULL;
   buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
   buf_info.flags = flags;

   VK_CHECK(vkCreateBuffer(ctx->device, &buf_info, NULL, &buffer.buf));

   // Look for suitable memory heap
   vkGetBufferMemoryRequirements(ctx->device, buffer.buf, &buffer.mem_reqs);

   VkMemoryAllocateInfo alloc_info;
   alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
   alloc_info.pNext = NULL;
   alloc_info.allocationSize = buffer.mem_reqs.size;
   bool result =
      vkdf_memory_type_from_properties(ctx, buffer.mem_reqs.memoryTypeBits,
                                       mem_props,
                                       &alloc_info.memoryTypeIndex);
   assert(result);
   buffer.mem_props = mem_props;

   // Allocate and bind memory
   VK_CHECK(vkAllocateMemory(ctx->device, &alloc_info, NULL, &buffer.mem));
   VK_CHECK(vkBindBufferMemory(ctx->device, buffer.buf, buffer.mem, 0));

   return buffer;
}

/**
 * If buf's memory is non-coherent, rounds up size to a multiple of
 * nonCoherentAtomSize, or to VK_WHOLE_SIZE if that would take us past the
 * end of buf's allocation. This is required so that the range we later
 * pass to vkFlushMappedMemoryRanges() (either directly, if we request the
 * same size for the mapping, or implicitly via VK_WHOLE_SIZE, which flushes
 * up to the end of the current mapping) is always valid: the end of the
 * mapping must be a multiple of nonCoherentAtomSize from the start of the
 * memory object or match the end of the memory object exactly.
 */
static VkDeviceSize
adjust_non_coherent_size(VkdfContext *ctx,
                         VkdfBuffer buf,
                         VkDeviceSize offset,
                         VkDeviceSize size)
{
   if ((buf.mem_props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ||
       size == VK_WHOLE_SIZE) {
      return size;
   }

   VkDeviceSize atom_size = ctx->phy_device_props.limits.nonCoherentAtomSize;

   assert(offset % atom_size == 0);

   VkDeviceSize aligned_size = ALIGN(size, atom_size);
   if (offset + aligned_size >= buf.mem_reqs.size)
      aligned_size = VK_WHOLE_SIZE;

   return aligned_size;
}

void
vkdf_buffer_map(VkdfContext *ctx,
                VkdfBuffer buf,
                VkDeviceSize offset,
                VkDeviceSize size,
                void **ptr)
{
   size = adjust_non_coherent_size(ctx, buf, offset, size);
   vkdf_memory_map(ctx, buf.mem, offset, size, ptr);
}

void
vkdf_buffer_unmap(VkdfContext *ctx,
                  VkdfBuffer buf,
                  VkDeviceSize offset,
                  VkDeviceSize size)
{
   size = adjust_non_coherent_size(ctx, buf, offset, size);
   vkdf_memory_unmap(ctx, buf.mem, buf.mem_props, offset, size);
}

void
vkdf_buffer_map_and_fill(VkdfContext *ctx,
                         VkdfBuffer buf,
                         VkDeviceSize offset,
                         VkDeviceSize size,
                         const void *data)
{
   assert(buf.mem_props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   assert(buf.mem_reqs.size >= offset + size);

   void *mapped_memory;
   vkdf_buffer_map(ctx, buf, offset, size, &mapped_memory);

   memcpy(mapped_memory, data, size);

   vkdf_buffer_unmap(ctx, buf, offset, size);
}

void
vkdf_buffer_map_and_fill_elements(VkdfContext *ctx,
                                  VkdfBuffer buf,
                                  VkDeviceSize offset,
                                  uint32_t num_elements,
                                  uint32_t element_size,
                                  uint32_t src_stride,
                                  uint32_t dst_stride,
                                  const void *data)
{
   assert(buf.mem_props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   assert(dst_stride >= element_size);

   VkDeviceSize size = num_elements * dst_stride;
   assert(buf.mem_reqs.size >= offset + size);

   void *mapped_memory;
   vkdf_buffer_map(ctx, buf, offset, size, &mapped_memory);

   VkDeviceSize dst_offset = 0;
   VkDeviceSize src_offset = 0;
   for (uint32_t i = 0; i < num_elements; i++) {
      memcpy((uint8_t *)mapped_memory + dst_offset,
             (uint8_t *)data + src_offset, element_size);
      dst_offset += dst_stride;
      src_offset += src_stride;
   }

   vkdf_buffer_unmap(ctx, buf, offset, size);
}

void
vkdf_destroy_buffer(VkdfContext *ctx, VkdfBuffer *buf)
{
   vkDestroyBuffer(ctx->device, buf->buf, NULL);
   vkFreeMemory(ctx->device, buf->mem, NULL);
}
