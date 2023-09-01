<template>
  <div id="app">
    <router-view :key="key"/>
  </div>
</template>

<script>
export default {
  name: 'App',
  data () {
    return {
    }
  },
  computed: {
    key () {
      return this.$route.path
    }
  }
}
</script>

<style>
</style>
