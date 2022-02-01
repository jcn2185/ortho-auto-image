<template>
  <span class="hello">
    <h1>{{ msg }}</h1>
    <div id="image-boundary" style="display: inline-block; position: relative;" ref="image_boundary" @drop="dropEvent($event)" @dragover="dragOver" draggable="false">
      <img id="vue-image" alt="Vue logo" src="../assets/logo.png" @click="handleClick($event)" draggable="false">
    </div>
    <h1> {{coordinates }}</h1>
  </span>
</template>

<script>
export default {
  name: 'ImageViewer',
  props: {
    msg: String,
    coordCount: {
      type: Number,
      default: 6
    },
    lockPoints: {
      type: Boolean,
      default: false
    },
    srcPath: {
      type: String,
      default: '../assets/logo.png'
    }
  },
  data: function () {
    var dotSize = 10
    var cursorOffset = dotSize * 0.5

    return {
      coordinates: '<none>',
      activeDrag: null,
      dotSize: 10,
      cursorOffset: cursorOffset
    }
  },
  methods: {

    dragStart ($event) {
      this.activeDrag = $event.target
    },
    dragOver ($event) {
      $event.preventDefault()
      return false
    },
    dropEvent ($event) {
      if (this.lockPoints) { return }
      // Debug
      var parentItem = document.getElementById('image-boundary').getBoundingClientRect()

      var dot = this.activeDrag
      dot.style.top = $event.clientY - parentItem.top - this.cursorOffset + 'px'
      dot.style.left = $event.clientX - parentItem.left - this.cursorOffset + 'px'
      this.activeDrag = null
      $event.preventDefault()
      return false
    },
    /* Coords should be passed as percent of pixels */
    createDot (coords) {
      var parentItem = document.getElementById('image-boundary').getBoundingClientRect()
      console.log(parentItem.width)
      console.log(parentItem.height)

      var vueImage = document.getElementById('vue-image').getBoundingClientRect()
      console.log(vueImage)

      // Select parent div
      var parentDiv = this.$refs.image_boundary
      var div = document.createElement('div')
      div.className = 'base-coord'
      div.setAttribute('draggable', true)
      div.style.position = 'absolute'
      div.style.top = (coords.y * vueImage.height) + 'px' // offset this to get it at the cursor point
      div.style.left = (coords.x * vueImage.width) + 'px' // offset this to get it at the cursor point
      div.style.width = this.dotSize + 'px'
      div.style.height = this.dotSize + 'px'
      div.style.backgroundColor = '#0066FF'

      // Drop event function
      if (!this.lockPoints) { div.addEventListener('dragstart', this.dragStart, false) }

      parentDiv.appendChild(div)
    },
    drawDot ($event) {
      this.createDot({
        x: 0.5,
        y: 0.5
      })

      // const mouseX = $event.offsetX
      // const mouseY = $event.offsetY
      // console.log(mouseX + ' ' + mouseY)

      // // Select parent div
      // var parentDiv = this.$refs.image_boundary
      // var div = document.createElement('div')
      // div.className = 'base-coord'
      // div.setAttribute('draggable', true)
      // div.style.position = 'absolute'
      // div.style.top = mouseY - this.cursorOffset + 'px' // offset this to get it at the cursor point
      // div.style.left = mouseX - this.cursorOffset + 'px' // offset this to get it at the cursor point
      // div.style.width = this.dotSize + 'px'
      // div.style.height = this.dotSize + 'px'
      // div.style.backgroundColor = '#0066FF'

      // // Drop event function
      // div.addEventListener('dragstart', this.dragStart, false)

      // parentDiv.appendChild(div)
    },
    handleClick ($event) {
      this.coordinates = { x: $event.offsetX, y: $event.offsetY }
      this.drawDot($event)
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h3 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}

.base-coord {
  background-color: #66FF00
}

#image-boundary {
  padding: 0;
  margin: 0;
  border-collapse: collapse;
}

#image-boundary img{
  display: block;
}

</style>
