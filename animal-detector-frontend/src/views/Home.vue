<template>
  <div class="home">
    <div class="menu-grid">
      <MenuItem
        v-for="item in menuItems"
        :key="item.id"
        :icon="item.icon"
        :text="item.text"
        @click="navigateTo(item.route)"
      />
    </div>
  </div>
</template>

<script>
import MenuItem from '../components/MenuItem.vue'
import { useRouter } from 'vue-router'

export default {
  name: 'Home',
  components: {
    MenuItem
  },
  setup() {
    const router = useRouter()

    const menuItems = [
      { id: 9, icon: '1', text: 'Test model', route: '/testing' },
      { id: 10, icon: '2', text: 'Evaluate model performance', route: '/testing' }
    ]

    const navigateTo = (route) => {
      router.push(route)
    }

    return {
      menuItems,
      navigateTo
    }
  }
}
</script>

<style scoped>
.home-container {
  height: calc(100vh - 80px);
  overflow: auto;
  padding: 20px;
}

.menu-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  height: 100%;
}
</style>