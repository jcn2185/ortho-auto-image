<template>
  <div class="centered">
    <div class="container">
      <grid-layout
        :layout.sync="layout"
        :col-num="9"
        :row-height="30"
        :is-draggable="false"
        :is-resizable="false"
        :is-mirrored="false"
        :margin="[10, 10]"
        :use-css-transforms="true"
      >
        <grid-item
          v-for="item in layout"
          :x="item.x"
          :y="item.y"
          :w="item.w"
          :h="item.h"
          :i="item.i"
          :key="item.i"
        >
          <img :src="item.url" />
        </grid-item>
      </grid-layout>
    </div>
    <div class="container-drag-drop" @dragover.prevent @drop.prevent>
      <div class="drop-area" @drop="handleFileDrop">
        <input
          type="file"
          name="file-input"
          multiple="True"
          @change="handleFileInput"
        />
        <ul>
          <li v-for="file in valid_files">
            {{ file.name }} {{ file.classification }}
          </li>
        </ul>
      </div>
      drag & drop
    </div>
  </div>
</template>

<script>
// @ is an alias to /src
import VueGridLayout from "vue-grid-layout";
const axios = require("axios").default;

export default {
  name: "about",
  components: {
    GridLayout: VueGridLayout.GridLayout,
    GridItem: VueGridLayout.GridItem,
  },
  data() {
    let eo = { w: 2, h: 8 }; // extraoral
    let ioo = { x: 0, y: eo.h, w: 2.25, h: 5 }; // intraoral
    let lio = { x: 0, y: eo.h + ioo.h, w: eo.w, h: 4 }; // left introral
    let ceph = { x: 3 * eo.w, y: 0, w: 3, h: 10 };
    let pan = { x: ceph.x, y: ceph.y + ceph.h, w: ceph.w, h: 5 };
    return {
      // Build grid lookup based on items
      layout: [
        { x: 0, y: 0, w: eo.w, h: eo.h, i: "1" },
        { x: 1 * eo.w, y: 0, w: eo.w, h: eo.h, i: "2" },
        { x: 2 * eo.w, y: 0, w: eo.w, h: eo.h, i: "3" },
        { x: ioo.x, y: ioo.y, w: ioo.w, h: ioo.h, i: "4" },
        { x: ioo.x + ioo.w + 1.5, y: ioo.y, w: ioo.w, h: ioo.h, i: "5" },
        { x: lio.x, y: lio.y, w: lio.w, h: lio.h, i: "6" },
        { x: lio.x + lio.w, y: lio.y, w: lio.w, h: lio.h, i: "7" },
        { x: lio.x + 2 * lio.w, y: lio.y, w: lio.w, h: lio.h, i: "8" },
        { x: ceph.x, y: ceph.y, w: ceph.w, h: ceph.h, i: "9" },
        { x: pan.x, y: pan.y, w: pan.w, h: pan.h, i: "10" },
      ],
      valid_files: [],
    };
  },
  methods: {
    addItem: async function(tmp_files) {
      // create form data to upload them
      let formData = new FormData();

      // Okay, so we have a list of files
      // We should now try to use axios and submit
      tmp_files.forEach((f) => {
        // See if file already exists

        formData.append(f.name, f);
      });

      var result = await axios.post("/api/image-classifier", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      this.valid_files = result.data;
      result.data.forEach((r) => {
        let idx = this.layout.findIndex((x) => x.i == r.classification);
        if (idx > -1) {
          let item = this.layout[idx];
          item.url = r.url;
          this.$set(this.layout, idx, item);
        } else {
          console.log("Unable to find classification: ", r.classification);
        }
      });
    },
    handleFileDrop(e) {
      console.log("in file drop");
      let files = e.dataTransfer.files;
      if (!files) return;
      this.addItem(files);
    },
    handleFileInput(e) {
      console.log("In file input");
      let files = e.target.files;
      if (!files) return;
      this.addItem(files);
    },
  },
};
</script>

<style>
img {
  width: 100%;
  height: 100%;
}
.centered {
  display: flex;
  justify-content: center;
}
.container {
  width: 1200px;
}
.container .vue-grid-item.vue-grid-placeholder {
  background: green;
}
.vue-grid-layout {
  background: #eee;
}
.vue-grid-item:not(.vue-grid-placeholder) {
  background: #ccc;
  border: 1px solid black;
}
.vue-grid-item .resizing {
  opacity: 0.9;
}
.vue-grid-item .static {
  background: #cce;
}
.vue-grid-item .text {
  font-size: 24px;
  text-align: center;
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  right: 0;
  margin: auto;
  height: 100%;
  width: 100%;
}
.vue-grid-item .no-drag {
  height: 100%;
  width: 100%;
}
.vue-grid-item .minMax {
  font-size: 12px;
}
.vue-grid-item .add {
  cursor: pointer;
}
.vue-draggable-handle {
  position: absolute;
  width: 20px;
  height: 20px;
  top: 0;
  left: 0;
  background: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'><circle cx='5' cy='5' r='5' fill='#999999'/></svg>")
    no-repeat;
  background-position: bottom right;
  padding: 0 8px 8px 0;
  background-repeat: no-repeat;
  background-origin: content-box;
  box-sizing: border-box;
  cursor: pointer;
}
</style>
