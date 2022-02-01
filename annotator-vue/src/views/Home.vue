<template>
  <div class="home">
    <img alt="Vue logo" src="../assets/logo.png">
    <table>
      <tr>
      <td>
        <ImageViewer msg="Welcome to Your Vue.s App" :lockPoints='true'/>
      </td>
      <td>
        <ImageViewer msg="Expected label" :lockPoints='true' srcPath="../assets/logo.png"/>
      </td>
      </tr>
      </table>
  </div>
</template>

<script>
// @ is an alias to /src
import ImageViewer from '@/components/ImageViewer.vue'

export default {
  name: 'Home',
  components: {
    ImageViewer
  }
}
</script>
