<template>
  <div class="logo-container">
      <img src="/logo.png" alt="Logo" class="logo" />
    </div>
  <div class="home">
    <div class="menu-container">
      <button
        v-for="item in menuItems"
        :key="item.id"
        class="menu-button"
        @click="navigateTo(item.route)"
      >
        {{ item.text }}
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'

export default {
  name: 'Home',
  setup() {
    const router = useRouter()

    const menuItems = [
      { id: 9, text: 'Test model', route: '/testing' },
      { id: 10, text: 'Evaluate model performance', route: '/testing' }
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
.home {
  width: 100%;
  min-height: 100vh;
  padding: 2rem 20px;
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #8D8A33 0%, #8D8A33 100%);
  box-sizing: border-box;
  position: relative;
}

.logo-container {
  position: absolute;
  top: 1px;
  right: 1px;
  z-index: 10;
}

.logo {
  height: 300px;
}

.menu-container {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  align-items: center;
  width: 100%;
  max-width: 500px;
}

.menu-button {
  background: linear-gradient(135deg, #F0AB0F 0%, #F0AB0F 100%);
  color: #563008;
  border: none;
  border-radius: 25px;
  padding: 15px 40px;
  font-size: 2rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  width: 100%;
  max-width: 500px;
  min-height: 50px;
}

.menu-button:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
  background: linear-gradient(135deg, #fb8c00 0%, #ef6c00 100%);
}

.menu-button:active {
  transform: translateY(-1px);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

</style>