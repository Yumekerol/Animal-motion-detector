<template>
  <header class="header">
    <div class="header-content">
      <div class="logo-container">
        <img src="/logo.png" alt="Logo" class="logo" />
      </div>
    </div>
  </header>
</template>

<script>
export default {
  name: 'Header'
}
</script>

<style scoped>
.header {
  background-color: #8D8A33;
  color: white;
  padding: 1rem 10px;
  width: 100vw;
  left: 0;
  right: 0;
  margin: 0;
}

.header-content {
  max-width: 100%;
  margin: 0;
  padding: 0;
  width: 100%;
  display: flex;
  justify-content: flex-end;
}

.logo-container {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.logo {
  height: 300px;
}
</style>