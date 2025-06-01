<template>
  <button class="menu-item" @click="$emit('click')">
    <span class="icon">{{ icon }}</span>
    <span class="text">{{ text }}</span>
  </button>
</template>

<script>
export default {
  name: 'MenuItem',
  props: {
    icon: {
      type: String,
      required: true
    },
    text: {
      type: String,
      required: true
    }
  }
}
</script>

<style scoped>
.menu-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  width: 100%;
  background: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.menu-item:hover {
  background: #e0e0e0;
  transform: translateY(-2px);
}

.icon {
  font-size: 1.5rem;
  font-weight: bold;
}

.text {
  font-size: 1rem;
}
</style>