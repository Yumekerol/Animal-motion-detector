import { createRouter, createWebHistory } from 'vue-router'
import Home from './views/Home.vue'
import DatasetView from './views/DatasetView.vue'
import TrainingView from './views/TrainingView.vue'
import TestingView from './views/TestingView.vue'

const routes = [
  { path: '/', component: Home },
  { path: '/datasets', component: DatasetView },
  { path: '/training', component: TrainingView },
  { path: '/testing', component: TestingView }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router