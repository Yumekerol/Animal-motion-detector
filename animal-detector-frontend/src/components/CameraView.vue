<template>
  <div class="camera-container">
    <div class="camera-box">
      <video ref="video" autoplay class="camera-feed"></video>
      <div class="controls">
        <button @click="startCamera" class="control-button">Start</button>
        <button @click="stopCamera" class="control-button">Stop</button>
        <button @click="captureImage" class="control-button">Capture</button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CameraView',
  data() {
    return {
      stream: null
    }
  },
  methods: {
    async startCamera() {
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ video: true })
        this.$refs.video.srcObject = this.stream
      } catch (err) {
        console.error("Error accessing camera:", err)
        alert("Could not access the camera")
      }
    },
    stopCamera() {
      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop())
        this.$refs.video.srcObject = null
      }
    },
    captureImage() {
      const video = this.$refs.video
      const canvas = document.createElement('canvas')
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)

      const imageData = canvas.toDataURL('image/png')
      this.$emit('captured', imageData)
    }
  },
  beforeUnmount() {
    this.stopCamera()
  }
}
</script>

<style scoped>
.camera-container {
  display: flex;
  justify-content: center;
  margin-top: 2rem;
}

.camera-box {
  background: white;
  border-radius: 8px;
  padding: 1rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 800px;
}

.camera-feed {
  width: 100%;
  border-radius: 4px;
  background: #333;
  aspect-ratio: 16/9;
}

.controls {
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
  justify-content: center;
}

.control-button {
  padding: 0.5rem 1rem;
  background: #2c3e50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.2s;
}

.control-button:hover {
  background: #1a252f;
}
</style>