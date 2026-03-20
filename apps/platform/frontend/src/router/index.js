import { createRouter, createWebHistory } from 'vue-router'
import EncodingView from '../views/EncodingView.vue'
import AnnotationView from '../views/AnnotationView.vue'
import ReviewTasksView from '../views/ReviewTasksView.vue'
import ReviewDetailView from '../views/ReviewDetailView.vue'

const routes = [
  {
    path: '/',
    redirect: '/encoding'
  },
  {
    path: '/encoding',
    name: 'encoding',
    component: EncodingView,
    meta: { title: '编码' }
  },
  {
    path: '/annotation',
    name: 'annotation',
    component: AnnotationView,
    meta: { title: '标注' }
  },
  {
    path: '/review',
    name: 'review',
    component: ReviewTasksView,
    meta: { title: '代码审核' }
  },
  {
    path: '/review/:id',
    name: 'review-detail',
    component: ReviewDetailView,
    meta: { title: '任务详情' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  document.title = `${to.meta.title || '材料智能处理平台'} - 材料智能处理平台`
  next()
})

export default router
