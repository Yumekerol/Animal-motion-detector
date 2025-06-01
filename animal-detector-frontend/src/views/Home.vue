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
      { id: 1, icon: '1', text: 'Download and merge datasets', route: '/datasets' },
      { id: 2, icon: '2', text: 'Validate merged dataset', route: '/datasets' },
      { id: 3, icon: '3', text: 'Train model', route: '/training' },
      { id: 9, icon: '4', text: 'Test model', route: '/testing' },
      { id: 10, icon: '5', text: 'Evaluate model performance', route: '/testing' }
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
  height: calc(100vh - 80px); /* Altura total menos o header */
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