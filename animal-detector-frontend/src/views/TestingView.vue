<template>
  <div class="logo-container">
      <img src="/logo.png" alt="Logo" class="logo" />
    </div>
  <div class="testing-view">

    <h2>Model Testing</h2>

    <div class="test-options">
      <button
        v-for="option in testOptions"
        :key="option.id"
        class="test-option-button"
        :class="{ selected: selectedOption === option.id }"
        @click="selectOption(option.id)"
      >
        {{ option.name }}
      </button>
    </div>

    <div class="content-area">
      <div class="main-content">
        <div v-if="selectedOption === 1" class="webcam-container">
          <CameraView @captured="handleCapture" />
        </div>

        <div v-if="selectedOption === 2" class="image-upload">
          <input type="file" id="image-upload" accept="image/*" @change="handleImageUpload" />
          <label for="image-upload" class="upload-button">
            Upload Image
          </label>
          <div v-if="uploadedImage" class="image-preview">
            <img :src="uploadedImage" alt="Uploaded" />
          </div>
        </div>

        <div v-if="selectedOption === 3" class="video-upload">
          <input type="file" id="video-upload" accept="video/*" @change="handleVideoUpload" />
          <label for="video-upload" class="upload-button">
            Upload Video
          </label>
          <div v-if="uploadedVideo" class="video-preview">
            <video :src="uploadedVideo" controls></video>
          </div>
        </div>

        <button
          @click="startTesting"
          class="test-button"
          :disabled="!selectedOption || testing"
        >
          {{ testing ? 'Processing...' : 'Start Testing' }}
        </button>
      </div>

    </div>

    <div v-if="results" class="results">
      <h3>Detection Results</h3>
      <div class="result-image" v-if="results.image">
        <img :src="results.image" alt="Detection Result" />
      </div>
    </div>
  </div>
</template>

<script>
import { ref, nextTick } from 'vue'
import CameraView from '../components/CameraView.vue'
import axios from 'axios'

export default {
  name: 'TestingView',
  components: {
    CameraView
  },
  setup() {
    const testOptions = [
      { id: 1, name: 'Webcam' },
      { id: 2, name: 'Image' },
      { id: 3, name: 'Video' }
    ]

    const selectedOption = ref(null)
    const testing = ref(false)
    const uploadedImage = ref(null)
    const uploadedVideo = ref(null)
    const capturedImage = ref(null)
    const results = ref(null)
    const chartCanvas = ref(null)

    const selectOption = (id) => {
      selectedOption.value = id
      results.value = null
    }

    const handleCapture = (imageData) => {
      capturedImage.value = imageData
    }

    const handleImageUpload = (event) => {
      const file = event.target.files[0]
      if (file) {
        const reader = new FileReader()
        reader.onload = (e) => {
          uploadedImage.value = e.target.result
        }
        reader.readAsDataURL(file)
      }
    }

    const handleVideoUpload = (event) => {
      const file = event.target.files[0]
      if (file) {
        uploadedVideo.value = URL.createObjectURL(file)
      }
    }

    const drawChart = async () => {
      await nextTick()
      if (!chartCanvas.value || !results.value) return

      const canvas = chartCanvas.value
      const ctx = canvas.getContext('2d')
      const centerX = canvas.width / 2
      const centerY = canvas.height / 2
      const radius = 50

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const classCounts = results.value.classCounts
      const total = Object.values(classCounts).reduce((sum, count) => sum + count, 0)

      if (total === 0) return

      const colors = {
        'Cats': '#4285F4', // Blue
        'Dogs': '#EA4335'  // Red
      }

      let currentAngle = -Math.PI / 2 // Start from top

      Object.entries(classCounts).forEach(([className, count]) => {
        const sliceAngle = (count / total) * 2 * Math.PI

        ctx.beginPath()
        ctx.moveTo(centerX, centerY)
        ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle)
        ctx.closePath()
        ctx.fillStyle = colors[className] || '#999'
        ctx.fill()

        currentAngle += sliceAngle
      })
    }

    const startTesting = async () => {
      if (!selectedOption.value) return

      testing.value = true
      results.value = null

      try {
        let formData = new FormData()
        let endpoint = ''

        if (selectedOption.value === 1 && capturedImage.value) {
          const blob = dataURLtoBlob(capturedImage.value)
          formData.append('image', blob, 'capture.png')
          endpoint = '/api/test/webcam'
        } else if (selectedOption.value === 2 && uploadedImage.value) {
          const blob = dataURLtoBlob(uploadedImage.value)
          formData.append('image', blob, 'upload.png')
          endpoint = '/api/test/image'
        } else if (selectedOption.value === 3 && uploadedVideo.value) {
          const response = await fetch(uploadedVideo.value)
          const blob = await response.blob()
          formData.append('video', blob, 'video.mp4')
          endpoint = '/api/test/video'
        } else {
          throw new Error('No media selected for testing')
        }

        const response = await axios.post(`http://localhost:5000${endpoint}`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })

        results.value = response.data
        await drawChart()
      } catch (error) {
        console.error('Testing error:', error)
        // Mock data for demonstration
        results.value = {
          classCounts: { 'Cats': 7, 'Dogs': 3 },
          image: null
        }
        await drawChart()
      } finally {
        testing.value = false
      }
    }

    const dataURLtoBlob = (dataURL) => {
      const arr = dataURL.split(',')
      const mime = arr[0].match(/:(.*?);/)[1]
      const bstr = atob(arr[1])
      let n = bstr.length
      const u8arr = new Uint8Array(n)
      while (n--) {
        u8arr[n] = bstr.charCodeAt(n)
      }
      return new Blob([u8arr], { type: mime })
    }

    return {
      testOptions,
      selectedOption,
      testing,
      uploadedImage,
      uploadedVideo,
      capturedImage,
      results,
      chartCanvas,
      selectOption,
      handleCapture,
      handleImageUpload,
      handleVideoUpload,
      startTesting
    }
  }
}
</script>

<style scoped>
.testing-view {
  width: 100%;
  max-width: 85%;
  justify-content: center;
  align-items: center;
  margin: 0;
  padding: 1rem 20px;
  box-sizing: border-box;
  background: linear-gradient(135deg, #8D8A33 0%, #8D8A33 100%);
  min-height: 100vh;
  position: relative;
}

.logo-container {
  position: absolute;
  top: 1px;
  right: 1px;
  z-index: 10;
}

.logo {
  height: 300px;
}

h2 {
  text-align: center;
  margin-bottom: 1.5rem;
  margin-top: 1rem;
  color: #563008;
  font-size: 2rem;
  font-weight: bold;
}

.test-options {
  display: flex;
  justify-content: center;
  gap: 2rem;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
}

.test-option-button {
  background: linear-gradient(135deg, #F0AB0F 0%, #F0AB0F 100%);
  color: #563008;
  border: none;
  border-radius: 25px;
  padding: 12px 30px;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  min-width: 120px;
}

.test-option-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

.test-option-button.selected {
  background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
  transform: translateY(-2px);
}

.content-area {
  display: flex;
  gap: 2rem;
  align-items: flex-start;
}

.main-content {
  flex: 1;
  background: white;
  border-radius: 15px;
  padding: 2rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  min-height: 400px;
}

.webcam-container,
.image-upload,
.video-upload {
  margin: 1rem 0;
}

input[type="file"] {
  display: none;
}

.upload-button {
  display: inline-block;
  padding: 12px 24px;
  background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
  color: white;
  border-radius: 25px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  font-weight: 500;
}

.upload-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

.image-preview,
.video-preview {
  margin-top: 1rem;
  text-align: center;
}

.image-preview img {
  max-width: 100%;
  max-height: 300px;
  border-radius: 10px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.video-preview video {
  max-width: 100%;
  max-height: 300px;
  border-radius: 10px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.test-button {
  padding: 12px 30px;
  background: linear-gradient(135deg, #FF5722 0%, #D84315 100%);
  color: white;
  border: none;
  border-radius: 25px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 1rem;
  font-weight: 500;
  margin-top: 2rem;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.test-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

.test-button:disabled {
  background: #95a5a6;
  cursor: not-allowed;
  transform: none;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.results {
  background: white;
  padding: 2rem;
  border-radius: 15px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  margin-top: 2rem;
  width: 100%;
}

.results h3 {
  margin-bottom: 1rem;
  color: #5d4037;
  font-size: 1.5rem;
}

.result-image {
  text-align: center;
}

.result-image img {
  max-width: 100%;
  border-radius: 10px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

</style>