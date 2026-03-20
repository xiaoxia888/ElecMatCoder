<template>
  <div class="app">
    <!-- 顶部导航栏 -->
    <header class="header">
      <div class="header-left">
        <h1 class="logo">材料智能处理平台</h1>
        <nav class="nav-switch">
          <router-link 
            v-for="nav in navItems" 
            :key="nav.path"
            :to="nav.path"
            class="nav-btn"
            active-class="active"
          >
            {{ nav.label }}
          </router-link>
        </nav>
      </div>
      <div class="header-right"></div>
    </header>

    <!-- 路由内容区 -->
    <main class="main">
      <router-view />
    </main>

    <!-- Toast 消息 -->
    <Toast ref="toastRef" />
  </div>
</template>

<script setup>
import { ref, provide } from 'vue'
import Toast from './components/Toast.vue'

// 导航配置（编码在前）
const navItems = [
  { path: '/encoding', label: '编码' },
  { path: '/annotation', label: '标注' },
  { path: '/review', label: '代码审核' }
]

const toastRef = ref(null)

// 提供全局方法
provide('showToast', (msg, type) => {
  toastRef.value?.show(msg, type)
})
</script>

<style scoped>
.app {
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* 顶部导航 */
.header {
  height: 56px;
  background: var(--bg-primary);
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  flex-shrink: 0;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 24px;
}

.logo {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
}

.nav-switch {
  display: flex;
  gap: 4px;
  background: var(--bg-tertiary);
  padding: 4px;
  border-radius: 6px;
}

.nav-btn {
  padding: 6px 20px;
  border: none;
  background: transparent;
  color: var(--text-secondary);
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s;
  text-decoration: none;
}

.nav-btn:hover {
  color: var(--text-primary);
}

.nav-btn.active {
  background: var(--bg-primary);
  color: var(--primary);
  box-shadow: var(--shadow-sm);
}

/* 主内容区 */
.main {
  flex: 1;
  display: flex;
  overflow: hidden;
  width: 100%;
}

.main > :deep(*) {
  width: 100%;
}
</style>
