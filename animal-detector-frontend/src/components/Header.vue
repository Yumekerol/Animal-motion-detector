<template>
  <header class="header">
    <div class="logo-container">
      <img src="/logo.png" alt="Logo" class="logo" />
      <div class="logo-text">
        <h1>DETECT</h1>
        <p>Stray animals</p>
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
  background-color: #2c3e50;
  color: white;
  padding: 1rem 0;
  width: 100%;
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}


.logo-container {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.logo {
  height: 50px;
}

.logo-text h1 {
  font-size: 1.5rem;
  margin: 0;
}

.logo-text p {
  font-size: 0.9rem;
  margin: 0;
  opacity: 0.8;
}
</style>