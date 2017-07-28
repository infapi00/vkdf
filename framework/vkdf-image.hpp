#ifndef __VKDF_IMAGE_H__
#define __VKDF_IMAGE_H__

typedef struct {
   VkImage image;
   VkFormat format;
   VkDeviceMemory mem;
   VkMemoryRequirements mem_reqs;
   uint32_t mem_props;
   VkImageView view;
} VkdfImage;

VkdfImage
vkdf_create_image(VkdfContext *ctx,
                  uint32_t width,
                  uint32_t height,
                  uint32_t num_levels,
                  VkImageType image_type,
                  VkFormat format,
                  VkFormatFeatureFlags format_flags,
                  VkImageUsageFlags usage_flags,
                  uint32_t mem_props,
                  VkImageAspectFlags aspect_flags,
                  VkImageViewType image_view_type);

VkdfImage
vkdf_create_image_detailed(VkdfContext *ctx,
                           uint32_t width,
                           uint32_t height,
                           uint32_t num_levels,
                           VkImageType image_type,
                           VkFormat format,
                           VkFormatFeatureFlags format_flags,
                           VkImageUsageFlags usage_flags,
                           uint32_t mem_props,
                           VkImageAspectFlags aspect_flags,
                           VkImageViewType image_view_type,
                           VkImageLayout image_layout);

void
vkdf_destroy_image(VkdfContext *ctx, VkdfImage *image);

VkImageSubresourceRange
vkdf_create_image_subresource_range(VkImageAspectFlags aspect,
                                    uint32_t base_level,
                                    uint32_t level_count,
                                    uint32_t base_layer,
                                    uint32_t layer_count);

VkImageSubresourceLayers
vkdf_create_image_subresource_layers(VkImageAspectFlags aspect,
                                     uint32_t level,
                                     uint32_t base_layer,
                                     uint32_t layer_count);

VkImageCopy
vkdf_create_image_copy_region(VkImageSubresourceLayers src_subresource_layers,
                              uint32_t src_offset_x,
                              uint32_t src_offset_y,
                              uint32_t src_offset_z,
                              VkImageSubresourceLayers dst_subresource_layers,
                              uint32_t dst_offset_x,
                              uint32_t dst_offset_y,
                              uint32_t dst_offset_z,
                              uint32_t width,
                              uint32_t height,
                              uint32_t depth);

void
vkdf_image_set_layout(VkdfContext *ctx,
                      VkCommandBuffer cmd_buf,
                      VkImage image,
                      VkImageSubresourceRange subresource_range,
                      VkImageLayout old_layout,
                      VkImageLayout new_layout,
                      VkPipelineStageFlags src_stage_mask,
                      VkPipelineStageFlags dst_stage_mask);

void
vkdf_image_map_and_get(VkdfContext *ctx,
                       VkdfImage image,
                       VkDeviceSize offset,
                       VkDeviceSize size,
                       void *data);

void
vkdf_image_map_and_fill(VkdfContext *ctx,
                        VkdfImage image,
                        VkDeviceSize offset,
                        VkDeviceSize size,
                        const void *data);

#endif
