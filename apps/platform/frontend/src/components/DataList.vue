<template>
  <div class="data-list">
    <div 
      v-for="(item, index) in items"
      :key="index"
      class="data-item"
      :class="{ 
        active: currentIndex === index,
        annotated: isAnnotated(index),
        encoded: isEncoded(index)
      }"
      @click="$emit('select', index)"
    >
      <span class="item-index">#{{ index + 1 }}</span>
      <span class="item-text" :title="item.text">{{ item.text }}</span>
      <span class="item-status">
        <span v-if="isAnnotated(index)" class="status-dot annotated" title="已标注"></span>
        <span v-if="isEncoded(index)" class="status-dot encoded" title="已编码"></span>
      </span>
    </div>
    
    <div v-if="items.length === 0" class="empty-list">
      暂无数据
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  items: {
    type: Array,
    default: () => []
  },
  currentIndex: {
    type: Number,
    default: -1
  },
  annotations: {
    type: Object,
    default: () => ({})
  },
  encodings: {
    type: Object,
    default: () => ({})
  }
})

defineEmits(['select'])

function isAnnotated(index) {
  const ann = props.annotations[index]
  return ann && ann.tokens && ann.tokens.length > 0
}

function isEncoded(index) {
  const enc = props.encodings[index]
  return enc && enc.final_code
}
</script>

<style scoped>
.data-list {
  flex: 1;
  overflow-y: auto;
  min-height: 0;
}

.data-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.15s;
  margin-bottom: 4px;
}

.data-item:hover {
  background: var(--bg-tertiary);
}

.data-item.active {
  background: var(--primary-light);
  border-left: 3px solid var(--primary);
}

.item-index {
  font-size: 11px;
  color: var(--text-muted);
  min-width: 32px;
}

.item-text {
  flex: 1;
  font-size: 13px;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.item-status {
  display: flex;
  gap: 4px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-dot.annotated {
  background: var(--primary);
}

.status-dot.encoded {
  background: var(--success);
}

.empty-list {
  padding: 20px;
  text-align: center;
  color: var(--text-muted);
  font-size: 13px;
}
</style>
