<template>
  <div class="app">
    <main>
      <router-view />
    </main>
  </div>
</template>

<script>

export default {
  name: 'App'
}
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Itim&display=swap');

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  font-family: 'Itim', cursive;
}

html, body {
  width: 100vw;
  height: 100vh;
  overflow-x: hidden;
}

.app {
  width: 100vw;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #8D8A33 0%, #8D8A33 100%);
}

main {
  flex: 1;
  width: 100%;
  padding: 0;
  max-width: 100vw;
  overflow-x: hidden;
}
</style>