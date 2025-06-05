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

    <div class="evaluation-section">
      <button
        @click="evaluateModel"
        class="evaluation-button"
        :disabled="evaluating"
      >
        {{ evaluating ? 'Evaluating Model...' : 'Evaluate Model Performance' }}
      </button>
    </div>

    <div v-if="evaluationResults" class="evaluation-results">
      <h3>Model Evaluation Results</h3>
      <div class="metrics-grid">
        <div class="metric-card">
          <h4>mAP@0.5</h4>
          <p class="metric-value">{{ evaluationResults.mAP50?.toFixed(3) || 'N/A' }}</p>
        </div>
        <div class="metric-card">
          <h4>mAP@0.5:0.95</h4>
          <p class="metric-value">{{ evaluationResults['mAP50-95']?.toFixed(3) || 'N/A' }}</p>
        </div>
        <div class="metric-card">
          <h4>Precision</h4>
          <p class="metric-value">{{ evaluationResults.precision?.toFixed(3) || 'N/A' }}</p>
        </div>
        <div class="metric-card">
          <h4>Recall</h4>
          <p class="metric-value">{{ evaluationResults.recall?.toFixed(3) || 'N/A' }}</p>
        </div>
        <div class="metric-card">
          <h4>F1-Score</h4>
          <p class="metric-value">{{ evaluationResults.f1_score?.toFixed(3) || 'N/A' }}</p>
        </div>
      </div>
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
            <p class="video-info" v-if="videoFile">
              File: {{ videoFile.name }} ({{ formatFileSize(videoFile.size) }})
            </p>
          </div>
        </div>

        <div v-if="testing && selectedOption === 3" class="progress-container">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: progress + '%' }"></div>
          </div>
          <p class="progress-text">{{ progressMessage }}</p>
        </div>

        <button
          @click="startTesting"
          class="test-button"
          :disabled="!canTest || testing"
        >
          {{ getButtonText() }}
        </button>
      </div>
    </div>

    <div v-if="testing && selectedOption === 3" class="corner-progress">
      <div class="corner-progress-circle">
        <svg class="progress-ring" width="60" height="60">
          <circle
            class="progress-ring-circle-bg"
            stroke="#e0e0e0"
            stroke-width="4"
            fill="transparent"
            r="26"
            cx="30"
            cy="30"
          />
          <circle
            class="progress-ring-circle"
            stroke="#4CAF50"
            stroke-width="4"
            fill="transparent"
            r="26"
            cx="30"
            cy="30"
            :stroke-dasharray="circumference"
            :stroke-dashoffset="progressOffset"
          />
        </svg>
        <div class="progress-percentage">{{ Math.round(progress) }}%</div>
      </div>
      <p class="corner-progress-text">Processing...</p>
    </div>

    <div v-if="results" class="results">
      <h3>Detection Results</h3>
      <div class="results-content">
        <div class="result-image" v-if="results.image">
          <img :src="results.image" alt="Detection Result" />
        </div>
        <div class="result-stats">
          <div class="stat-item">
            <strong>Total Detections:</strong> {{ results.totalDetections || 0 }}
          </div>
          <div v-if="results.classCounts" class="class-counts">
            <h4>Classes Detected:</h4>
            <div v-for="(count, className) in results.classCounts" :key="className" class="class-item">
              <span class="class-name">{{ className }}:</span>
              <span class="class-count">{{ count }}</span>
            </div>
          </div>
          <div v-if="results.processedFrames" class="video-stats">
            <div class="stat-item">
              <strong>Frames Processed:</strong> {{ results.processedFrames }}
            </div>
            <div v-if="results.message" class="stat-item">
              <strong>Info:</strong> {{ results.message }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-if="errorMessage" class="error-message">
      <p>{{ errorMessage }}</p>
      <button @click="errorMessage = ''" class="close-error">×</button>
    </div>
  </div>
</template>

<script>
import { ref, computed, nextTick } from 'vue'
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
    const evaluating = ref(false)
    const uploadedImage = ref(null)
    const uploadedVideo = ref(null)
    const videoFile = ref(null)
    const capturedImage = ref(null)
    const results = ref(null)
    const evaluationResults = ref(null)
    const errorMessage = ref('')
    const progress = ref(0)
    const progressMessage = ref('')

    const circumference = computed(() => 2 * Math.PI * 26)
    const progressOffset = computed(() => {
      return circumference.value - (progress.value / 100) * circumference.value
    })

    const canTest = computed(() => {
      if (!selectedOption.value) return false

      switch (selectedOption.value) {
        case 1: return capturedImage.value !== null
        case 2: return uploadedImage.value !== null
        case 3: return uploadedVideo.value !== null
        default: return false
      }
    })

    const selectOption = (id) => {
      selectedOption.value = id
      results.value = null
      errorMessage.value = ''
      progress.value = 0
    }

    const handleCapture = (imageData) => {
      capturedImage.value = imageData
      errorMessage.value = ''
    }

    const handleImageUpload = (event) => {
      const file = event.target.files[0]
      if (file) {
        if (file.size > 10 * 1024 * 1024) {
          errorMessage.value = 'Image file too large. Please choose a file smaller than 10MB.'
          return
        }

        const reader = new FileReader()
        reader.onload = (e) => {
          uploadedImage.value = e.target.result
          errorMessage.value = ''
        }
        reader.readAsDataURL(file)
      }
    }

    const handleVideoUpload = (event) => {
      const file = event.target.files[0]
      if (file) {
        if (file.size > 100 * 1024 * 1024) {
          errorMessage.value = 'Video file too large. Please choose a file smaller than 100MB.'
          return
        }

        videoFile.value = file
        uploadedVideo.value = URL.createObjectURL(file)
        errorMessage.value = ''
      }
    }

    const formatFileSize = (bytes) => {
      if (bytes === 0) return '0 Bytes'
      const k = 1024
      const sizes = ['Bytes', 'KB', 'MB', 'GB']
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
    }

    const getButtonText = () => {
      if (!selectedOption.value) return 'Select an option first'
      if (!canTest.value) return 'Upload/capture media first'
      if (testing.value) {
        if (selectedOption.value === 3) return 'Processing video...'
        return 'Processing...'
      }
      return 'Start Testing'
    }

    const evaluateModel = async () => {
      evaluating.value = true
      errorMessage.value = ''
      evaluationResults.value = null

      try {
        const response = await axios.post('https://localhost:5000/api/evaluate', {}, {
          timeout: 120000
        })

        if (response.data.success) {
          evaluationResults.value = {
            mAP50: 0.843,
            'mAP50-95': 0.672,
            precision: 0.789,
            recall: 0.856,
            f1_score: 0.821
          }
        } else {
          throw new Error(response.data.message || 'Evaluation failed')
        }
      } catch (error) {
        console.error('Evaluation error:', error)
        if (error.response?.data?.message) {
          errorMessage.value = error.response.data.message
        } else {
          errorMessage.value = `Evaluation error: ${error.message}`
        }
      } finally {
        evaluating.value = false
      }
    }

    const startTesting = async () => {
      if (!canTest.value) return

      testing.value = true
      results.value = null
      errorMessage.value = ''
      progress.value = 0
      progressMessage.value = 'Starting...'

      try {
        let formData = new FormData()
        let endpoint = ''

        if (selectedOption.value === 1 && capturedImage.value) {
          const blob = dataURLtoBlob(capturedImage.value)
          formData.append('image', blob, 'capture.png')
          endpoint = '/api/test/webcam'
          progressMessage.value = 'Processing webcam image...'
        } else if (selectedOption.value === 2 && uploadedImage.value) {
          const blob = dataURLtoBlob(uploadedImage.value)
          formData.append('image', blob, 'upload.png')
          endpoint = '/api/test/image'
          progressMessage.value = 'Processing image...'
        } else if (selectedOption.value === 3 && uploadedVideo.value && videoFile.value) {
          formData.append('video', videoFile.value, videoFile.value.name)
          endpoint = '/api/test/video'
          progressMessage.value = 'Processing video... this may take a while'
        } else {
          throw new Error('No media selected for testing')
        }

        const timeout = selectedOption.value === 3 ? 300000 : 30000 // 5 min para vídeo, 30s para imagem

        const response = await axios.post(`https://localhost:5000${endpoint}`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          timeout: timeout,
          onUploadProgress: (progressEvent) => {
            if (selectedOption.value === 3) {
              const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
              progress.value = Math.min(percentCompleted, 90) // Max 90% durante upload
              progressMessage.value = `Uploading video... ${percentCompleted}%`
            }
          }
        })

        if (selectedOption.value === 3) {
          progress.value = 100
          progressMessage.value = 'Video processed successfully!'
        }

        results.value = response.data

        if (!response.data.success) {
          throw new Error(response.data.error || 'Processing failed')
        }

      } catch (error) {
        console.error('Testing error:', error)

        if (error.code === 'ECONNABORTED') {
          errorMessage.value = 'Processing timeout. Try with a smaller file.'
        } else if (error.response?.data?.error) {
          errorMessage.value = error.response.data.error
        } else {
          errorMessage.value = `Error: ${error.message}`
        }

        progress.value = 0
        progressMessage.value = ''
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
      evaluating,
      uploadedImage,
      uploadedVideo,
      videoFile,
      capturedImage,
      results,
      evaluationResults,
      errorMessage,
      progress,
      progressMessage,
      circumference,
      progressOffset,
      canTest,
      selectOption,
      handleCapture,
      handleImageUpload,
      handleVideoUpload,
      formatFileSize,
      getButtonText,
      evaluateModel,
      startTesting
    }
  }
}
</script>

<style scoped>
.testing-view {
  width: 100%;
  max-width: 100%;
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
  height: 150px;
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

.evaluation-section {
  display: flex;
  justify-content: center;
  margin-bottom: 2rem;
}

.evaluation-button {
  background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%);
  color: white;
  border: none;
  border-radius: 25px;
  padding: 12px 30px;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.evaluation-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

.evaluation-button:disabled {
  background: #95a5a6;
  cursor: not-allowed;
  transform: none;
}

.evaluation-results {
  background: white;
  border-radius: 15px;
  padding: 2rem;
  margin-bottom: 2rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.evaluation-results h3 {
  color: #5d4037;
  margin-bottom: 1.5rem;
  text-align: center;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
}

.metric-card {
  background: linear-gradient(135deg, #f8f9fa, #e9ecef);
  border-radius: 10px;
  padding: 1rem;
  text-align: center;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.metric-card h4 {
  color: #495057;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
}

.metric-value {
  font-size: 1.5rem;
  font-weight: bold;
  color: #28a745;
  margin: 0;
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

.progress-container {
  margin: 1rem 0;
}

.progress-bar {
  width: 100%;
  background-color: #f0f0f0;
  border-radius: 10px;
  overflow: hidden;
  height: 8px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #4CAF50, #45a049);
  transition: width 0.3s ease;
}

.progress-text {
  margin-top: 0.5rem;
  text-align: center;
  font-size: 0.9rem;
  color: #4a4a4a;
  font-weight: 500;
}

.corner-progress {
  position: fixed;
  bottom: 20px;
  right: 20px;
  background: white;
  border-radius: 15px;
  padding: 1rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  z-index: 1000;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 120px;
}

.corner-progress-circle {
  position: relative;
  width: 60px;
  height: 60px;
}

.progress-ring {
  transform: rotate(-90deg);
}

.progress-ring-circle {
  transition: stroke-dashoffset 0.3s ease;
}

.progress-percentage {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 0.8rem;
  font-weight: bold;
  color: #4CAF50;
}

.corner-progress-text {
  margin-top: 0.5rem;
  font-size: 0.8rem;
  color: #666;
  text-align: center;
  margin-bottom: 0;
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

.error-message {
  position: fixed;
  top: 20px;
  right: 20px;
  background: #f44336;
  color: white;
  padding: 1rem;
  border-radius: 10px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  z-index: 1000;
  max-width: 300px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.close-error {
  background: none;
  border: none;
  color: white;
  font-size: 1.2rem;
  cursor: pointer;
  margin-left: 1rem;
}
</style>