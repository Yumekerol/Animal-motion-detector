<template>
  <div class="dataset-view">
    <h2>Dataset Management</h2>

    <div class="actions">
      <button @click="downloadDatasets" class="action-button">
        Download and Merge Datasets
      </button>

      <button @click="validateDataset" class="action-button">
        Validate Merged Dataset
      </button>
    </div>

    <div v-if="loading" class="loading">
      Processing...
    </div>

    <div v-if="message" class="message" :class="{ success: isSuccess, error: !isSuccess }">
      {{ message }}
    </div>

    <div v-if="datasetInfo" class="dataset-info">
      <h3>Dataset Information</h3>
      <p>Number of classes: {{ datasetInfo.num_classes }}</p>
      <p>Classes: {{ datasetInfo.class_names.join(', ') }}</p>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import axios from 'axios'

export default {
  name: 'DatasetView',
  setup() {
    const loading = ref(false)
    const message = ref('')
    const isSuccess = ref(false)
    const datasetInfo = ref(null)

    const downloadDatasets = async () => {
      loading.value = true
      message.value = ''
      try {
        const response = await axios.post('http://localhost:5000/api/datasets/download')
        isSuccess.value = response.data.success
        message.value = response.data.message
        if (response.data.success) {
          datasetInfo.value = {
            num_classes: response.data.num_classes,
            class_names: response.data.class_names
          }
        }
      } catch (error) {
        isSuccess.value = false
        message.value = 'Error downloading datasets: ' + error.message
      } finally {
        loading.value = false
      }
    }

    const validateDataset = async () => {
      loading.value = true
      message.value = ''
      try {
        const response = await axios.get('http://localhost:5000/api/datasets/validate')
        isSuccess.value = response.data.success
        message.value = response.data.message
      } catch (error) {
        isSuccess.value = false
        message.value = 'Error validating dataset: ' + error.message
      } finally {
        loading.value = false
      }
    }

    return {
      loading,
      message,
      isSuccess,
      datasetInfo,
      downloadDatasets,
      validateDataset
    }
  }
}
</script>

<style scoped>
.dataset-view {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
}

h2 {
  margin-bottom: 2rem;
  color: #2c3e50;
}

.actions {
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
}

.action-button {
  padding: 0.75rem 1.5rem;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.2s;
}

.action-button:hover {
  background: #2980b9;
}

.loading {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 4px;
  margin-bottom: 1rem;
  text-align: center;
}

.message {
  padding: 1rem;
  border-radius: 4px;
  margin-bottom: 1rem;
}

.message.success {
  background: #d4edda;
  color: #155724;
}

.message.error {
  background: #f8d7da;
  color: #721c24;
}

.dataset-info {
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  margin-top: 2rem;
}

.dataset-info h3 {
  margin-bottom: 1rem;
  color: #2c3e50;
}

.dataset-info p {
  margin-bottom: 0.5rem;
}
</style>