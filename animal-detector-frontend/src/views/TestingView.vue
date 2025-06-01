<template>
  <div class="testing-view">
    <h2>Model Testing</h2>

    <div class="test-options">
      <div
        v-for="option in testOptions"
        :key="option.id"
        class="test-option"
        :class="{ selected: selectedOption === option.id }"
        @click="selectOption(option.id)"
      >
        <h3>{{ option.name }}</h3>
        <p>{{ option.description }}</p>
      </div>
    </div>

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

    <div v-if="results" class="results">
      <h3>Detection Results</h3>
      <div class="result-image" v-if="results.image">
        <img :src="results.image" alt="Detection Result" />
      </div>
      <div class="result-stats">
        <div v-for="(count, className) in results.classCounts" :key="className" class="stat-item">
          <span class="class-name">{{ className }}</span>
          <span class="class-count">{{ count }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import CameraView from '../components/CameraView.vue'
import axios from 'axios'

export default {
  name: 'TestingView',
  components: {
    CameraView
  },
  setup() {
    const testOptions = [
      { id: 1, name: 'Webcam', description: 'Test with live camera feed' },
      { id: 2, name: 'Image', description: 'Test with an image file' },
      { id: 3, name: 'Video', description: 'Test with a video file' }
    ]

    const selectedOption = ref(null)
    const testing = ref(false)
    const uploadedImage = ref(null)
    const uploadedVideo = ref(null)
    const capturedImage = ref(null)
    const results = ref(null)

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
      } catch (error) {
        console.error('Testing error:', error)
        alert('Error during testing: ' + error.message)
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
  max-width: 1000px;
  margin: 0 auto;
  padding: 2rem;
}

h2 {
  margin-bottom: 2rem;
  color: #2c3e50;
}

.test-options {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.test-option {
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  cursor: pointer;
  transition: all 0.2s;
}

.test-option:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.test-option.selected {
  border: 2px solid #3498db;
  background: #f0f8ff;
}

.test-option h3 {
  color: #2c3e50;
  margin-bottom: 0.5rem;
}

.test-option p {
  color: #7f8c8d;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
}

.webcam-container,
.image-upload,
.video-upload {
  margin: 2rem 0;
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

input[type="file"] {
  display: none;
}

.upload-button {
  display: inline-block;
  padding: 0.75rem 1.5rem;
  background: #3498db;
  color: white;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.2s;
}

.upload-button:hover {
  background: #2980b9;
}

.image-preview,
.video-preview {
  margin-top: 1rem;
}

.image-preview img {
  max-width: 100%;
  border-radius: 4px;
}

.video-preview video {
  max-width: 100%;
  border-radius: 4px;
}

.test-button {
  padding: 0.75rem 1.5rem;
  background: #e67e22;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.2s;
  font-size: 1rem;
  margin-bottom: 2rem;
}

.test-button:hover:not(:disabled) {
  background: #d35400;
}

.test-button:disabled {
  background: #95a5a6;
  cursor: not-allowed;
}

.results {
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  margin-top: 2rem;
}

.results h3 {
  margin-bottom: 1rem;
  color: #2c3e50;
}

.result-image img {
  max-width: 100%;
  border-radius: 4px;
  margin-bottom: 1rem;
}

.result-stats {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.stat-item {
  background: #f8f9fa;
  padding: 0.75rem;
  border-radius: 4px;
  display: flex;
  justify-content: space-between;
}

.class-name {
  font-weight: bold;
  color: #2c3e50;
}

.class-count {
  background: #3498db;
  color: white;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
}
</style>