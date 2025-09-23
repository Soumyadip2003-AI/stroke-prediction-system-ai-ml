// React Components and JavaScript for NeuroPredict
const { useState, useEffect, useRef } = React;

// Initialize particles.js background
document.addEventListener('DOMContentLoaded', function() {
    if (typeof particlesJS !== 'undefined') {
        particlesJS('particles-js', {
            particles: {
                number: { value: 80 },
                color: { value: '#667eea' },
                shape: { type: 'circle' },
                opacity: { value: 0.5 },
                size: { value: 3 },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: '#667eea',
                    opacity: 0.4,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 2,
                    direction: 'none',
                    random: false,
                    straight: false,
                    out_mode: 'out',
                    bounce: false
                }
            },
            interactivity: {
                detect_on: 'canvas',
                events: {
                    onhover: { enable: true, mode: 'repulse' },
                    onclick: { enable: true, mode: 'push' },
                    resize: true
                }
            },
            retina_detect: true
        });
    }
});

// Global state management
let currentStep = 1;
let assessmentData = {};

// Smooth scrolling function
function scrollToAssessment() {
    document.getElementById('assessment').scrollIntoView({ 
        behavior: 'smooth' 
    });
}

// Step navigation functions
function nextStep() {
    if (currentStep < 3) {
        document.getElementById(`step${currentStep}`).classList.add('hidden');
        currentStep++;
        document.getElementById(`step${currentStep}`).classList.remove('hidden');
        updateStepIndicators();
        updateNavigationButtons();
    }
}

function previousStep() {
    if (currentStep > 1) {
        document.getElementById(`step${currentStep}`).classList.add('hidden');
        currentStep--;
        document.getElementById(`step${currentStep}`).classList.remove('hidden');
        updateStepIndicators();
        updateNavigationButtons();
    }
}

function updateStepIndicators() {
    const dots = document.querySelectorAll('.step-dot');
    dots.forEach((dot, index) => {
        if (index + 1 <= currentStep) {
            dot.classList.remove('bg-gray-600');
            dot.classList.add('bg-blue-500');
        } else {
            dot.classList.remove('bg-blue-500');
            dot.classList.add('bg-gray-600');
        }
    });
}

function updateNavigationButtons() {
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    
    prevBtn.disabled = currentStep === 1;
    if (currentStep === 1) {
        prevBtn.classList.add('opacity-50', 'cursor-not-allowed');
    } else {
        prevBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    }
    
    if (currentStep === 3) {
        nextBtn.style.display = 'none';
    } else {
        nextBtn.style.display = 'flex';
    }
}

// Form data collection
function collectFormData() {
    assessmentData = {
        age: parseInt(document.getElementById('age').value),
        gender: document.getElementById('gender').value,
        ever_married: document.getElementById('ever_married').value,
        hypertension: document.querySelector('input[name="hypertension"]:checked').value,
        heart_disease: document.querySelector('input[name="heart_disease"]:checked').value,
        avg_glucose_level: parseFloat(document.getElementById('avg_glucose_level').value),
        bmi: parseFloat(document.getElementById('bmi').value),
        work_type: document.getElementById('work_type').value,
        residence_type: document.getElementById('residence_type').value,
        smoking_status: document.getElementById('smoking_status').value
    };
    return assessmentData;
}

// AI Prediction function (real API call)
async function predictStrokeRisk() {
    const formData = collectFormData();
    
    // Show loading overlay
    document.getElementById('loadingOverlay').classList.remove('hidden');
    
    try {
        // Call the backend API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Hide loading overlay
        document.getElementById('loadingOverlay').classList.add('hidden');
        
        // Show results
        displayAPIResults(result, formData);
        
    } catch (error) {
        console.error('Prediction error:', error);
        
        // Hide loading overlay
        document.getElementById('loadingOverlay').classList.add('hidden');
        
        // Fallback to simulated prediction
        const riskScore = calculateRiskScore(formData);
        const probability = riskScore / 100;
        displayResults(probability, formData);
        
        // Show error message
        alert('API connection failed. Using simulated prediction.');
    }
}

// Risk calculation (simplified simulation)
function calculateRiskScore(data) {
    let score = 0;
    
    // Age factor
    if (data.age > 65) score += 30;
    else if (data.age > 50) score += 15;
    else if (data.age > 30) score += 5;
    
    // Health conditions
    if (data.hypertension === 'Yes') score += 25;
    if (data.heart_disease === 'Yes') score += 20;
    
    // BMI factor
    if (data.bmi > 30) score += 15;
    else if (data.bmi > 25) score += 8;
    
    // Glucose factor
    if (data.avg_glucose_level > 126) score += 20;
    else if (data.avg_glucose_level > 100) score += 10;
    
    // Smoking factor
    if (data.smoking_status === 'smokes') score += 15;
    else if (data.smoking_status === 'formerly smoked') score += 8;
    
    // Gender factor (simplified)
    if (data.gender === 'Male') score += 5;
    
    return Math.min(score, 95); // Cap at 95%
}

// Display API results
function displayAPIResults(result, data) {
    const percentage = Math.round(result.risk_percentage);
    const riskCategory = result.risk_category;
    const riskColor = result.risk_color;
    
    // Update risk gauge
    document.getElementById('riskPercentage').textContent = `${percentage}%`;
    document.getElementById('riskCategory').textContent = riskCategory;
    
    // Update gauge visual
    const gaugeFill = document.getElementById('gaugeFill');
    const rotation = (percentage / 100) * 180; // 0-180 degrees
    gaugeFill.style.transform = `rotate(${rotation}deg)`;
    gaugeFill.style.borderColor = riskColor;
    
    // Update confidence
    document.getElementById('confidence').textContent = result.confidence;
    
    // Display health analysis from API
    displayHealthAnalysis(result.health_analysis);
    
    // Display recommendations from API
    displayRecommendations(result.recommendations);
    
    // Show results section
    document.getElementById('results').classList.remove('hidden');
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

// Display results (fallback)
function displayResults(probability, data) {
    const percentage = Math.round(probability * 100);
    const riskCategory = getRiskCategory(percentage);
    const riskColor = getRiskColor(percentage);
    
    // Update risk gauge
    document.getElementById('riskPercentage').textContent = `${percentage}%`;
    document.getElementById('riskCategory').textContent = riskCategory;
    
    // Update gauge visual
    const gaugeFill = document.getElementById('gaugeFill');
    const rotation = (percentage / 100) * 180; // 0-180 degrees
    gaugeFill.style.transform = `rotate(${rotation}deg)`;
    gaugeFill.style.borderColor = riskColor;
    
    // Update confidence
    const confidence = percentage > 70 ? 'High' : percentage > 40 ? 'Medium' : 'Low';
    document.getElementById('confidence').textContent = confidence;
    
    // Generate health analysis
    generateHealthAnalysis(data);
    
    // Generate recommendations
    generateRecommendations(data, percentage);
    
    // Show results section
    document.getElementById('results').classList.remove('hidden');
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

function getRiskCategory(percentage) {
    if (percentage < 20) return 'Low Risk';
    if (percentage < 50) return 'Moderate Risk';
    return 'High Risk';
}

function getRiskColor(percentage) {
    if (percentage < 20) return '#10B981'; // Green
    if (percentage < 50) return '#F59E0B'; // Yellow
    return '#EF4444'; // Red
}

function generateHealthAnalysis(data) {
    const analysisItems = document.getElementById('analysisItems');
    const items = [];
    
    // BMI analysis
    if (data.bmi > 30) {
        items.push(`
            <div class="flex items-center justify-between p-3 bg-red-900 bg-opacity-30 rounded-lg">
                <div class="flex items-center">
                    <i class="fas fa-exclamation-triangle text-red-400 mr-3"></i>
                    <span>Obesity (BMI: ${data.bmi})</span>
                </div>
                <span class="text-red-400 font-semibold">High Risk</span>
            </div>
        `);
    } else if (data.bmi > 25) {
        items.push(`
            <div class="flex items-center justify-between p-3 bg-yellow-900 bg-opacity-30 rounded-lg">
                <div class="flex items-center">
                    <i class="fas fa-exclamation-circle text-yellow-400 mr-3"></i>
                    <span>Overweight (BMI: ${data.bmi})</span>
                </div>
                <span class="text-yellow-400 font-semibold">Moderate Risk</span>
            </div>
        `);
    } else {
        items.push(`
            <div class="flex items-center justify-between p-3 bg-green-900 bg-opacity-30 rounded-lg">
                <div class="flex items-center">
                    <i class="fas fa-check-circle text-green-400 mr-3"></i>
                    <span>Healthy BMI (${data.bmi})</span>
                </div>
                <span class="text-green-400 font-semibold">Low Risk</span>
            </div>
        `);
    }
    
    // Glucose analysis
    if (data.avg_glucose_level > 126) {
        items.push(`
            <div class="flex items-center justify-between p-3 bg-red-900 bg-opacity-30 rounded-lg">
                <div class="flex items-center">
                    <i class="fas fa-exclamation-triangle text-red-400 mr-3"></i>
                    <span>Diabetic Range (${data.avg_glucose_level} mg/dL)</span>
                </div>
                <span class="text-red-400 font-semibold">High Risk</span>
            </div>
        `);
    } else if (data.avg_glucose_level > 100) {
        items.push(`
            <div class="flex items-center justify-between p-3 bg-yellow-900 bg-opacity-30 rounded-lg">
                <div class="flex items-center">
                    <i class="fas fa-exclamation-circle text-yellow-400 mr-3"></i>
                    <span>Prediabetic Range (${data.avg_glucose_level} mg/dL)</span>
                </div>
                <span class="text-yellow-400 font-semibold">Moderate Risk</span>
            </div>
        `);
    } else {
        items.push(`
            <div class="flex items-center justify-between p-3 bg-green-900 bg-opacity-30 rounded-lg">
                <div class="flex items-center">
                    <i class="fas fa-check-circle text-green-400 mr-3"></i>
                    <span>Normal Glucose (${data.avg_glucose_level} mg/dL)</span>
                </div>
                <span class="text-green-400 font-semibold">Low Risk</span>
            </div>
        `);
    }
    
    // Age analysis
    if (data.age > 65) {
        items.push(`
            <div class="flex items-center justify-between p-3 bg-red-900 bg-opacity-30 rounded-lg">
                <div class="flex items-center">
                    <i class="fas fa-exclamation-triangle text-red-400 mr-3"></i>
                    <span>Advanced Age (${data.age} years)</span>
                </div>
                <span class="text-red-400 font-semibold">High Risk</span>
            </div>
        `);
    } else if (data.age > 50) {
        items.push(`
            <div class="flex items-center justify-between p-3 bg-yellow-900 bg-opacity-30 rounded-lg">
                <div class="flex items-center">
                    <i class="fas fa-exclamation-circle text-yellow-400 mr-3"></i>
                    <span>Middle Age (${data.age} years)</span>
                </div>
                <span class="text-yellow-400 font-semibold">Moderate Risk</span>
            </div>
        `);
    }
    
    analysisItems.innerHTML = items.join('');
}

// Display health analysis from API
function displayHealthAnalysis(healthAnalysis) {
    const analysisItems = document.getElementById('analysisItems');
    const items = healthAnalysis.map(item => `
        <div class="flex items-center justify-between p-3 ${getAnalysisBgColor(item.color)} rounded-lg">
            <div class="flex items-center">
                <i class="${item.icon} ${getAnalysisTextColor(item.color)} mr-3"></i>
                <span>${item.title}</span>
            </div>
            <span class="${getAnalysisTextColor(item.color)} font-semibold">${item.risk_level}</span>
        </div>
    `).join('');
    
    analysisItems.innerHTML = items;
}

// Display recommendations from API
function displayRecommendations(recommendations) {
    const recommendationList = document.getElementById('recommendationList');
    const items = recommendations.map(rec => `
        <div class="flex items-start p-3 bg-blue-900 bg-opacity-30 rounded-lg">
            <i class="${rec.icon} text-blue-400 mr-3 mt-1"></i>
            <div>
                <div class="font-semibold">${rec.title}</div>
                <div class="text-sm text-gray-300">${rec.description}</div>
            </div>
        </div>
    `).join('');
    
    recommendationList.innerHTML = items;
}

// Helper functions for styling
function getAnalysisBgColor(color) {
    switch(color) {
        case 'red': return 'bg-red-900 bg-opacity-30';
        case 'yellow': return 'bg-yellow-900 bg-opacity-30';
        case 'green': return 'bg-green-900 bg-opacity-30';
        default: return 'bg-gray-900 bg-opacity-30';
    }
}

function getAnalysisTextColor(color) {
    switch(color) {
        case 'red': return 'text-red-400';
        case 'yellow': return 'text-yellow-400';
        case 'green': return 'text-green-400';
        default: return 'text-gray-400';
    }
}

function generateRecommendations(data, percentage) {
    const recommendationList = document.getElementById('recommendationList');
    const recommendations = [];
    
    // BMI recommendations
    if (data.bmi > 30) {
        recommendations.push(`
            <div class="flex items-start p-3 bg-blue-900 bg-opacity-30 rounded-lg">
                <i class="fas fa-dumbbell text-blue-400 mr-3 mt-1"></i>
                <div>
                    <div class="font-semibold">Weight Management</div>
                    <div class="text-sm text-gray-300">Work with healthcare providers on a comprehensive weight management plan to reduce BMI below 30.</div>
                </div>
            </div>
        `);
    } else if (data.bmi > 25) {
        recommendations.push(`
            <div class="flex items-start p-3 bg-blue-900 bg-opacity-30 rounded-lg">
                <i class="fas fa-running text-blue-400 mr-3 mt-1"></i>
                <div>
                    <div class="font-semibold">Exercise & Diet</div>
                    <div class="text-sm text-gray-300">Maintain a balanced diet and regular exercise to reach optimal BMI below 25.</div>
                </div>
            </div>
        `);
    }
    
    // Glucose recommendations
    if (data.avg_glucose_level > 126) {
        recommendations.push(`
            <div class="flex items-start p-3 bg-blue-900 bg-opacity-30 rounded-lg">
                <i class="fas fa-stethoscope text-blue-400 mr-3 mt-1"></i>
                <div>
                    <div class="font-semibold">Diabetes Management</div>
                    <div class="text-sm text-gray-300">Consult with an endocrinologist about diabetes management and consider a low-carb diet.</div>
                </div>
            </div>
        `);
    } else if (data.avg_glucose_level > 100) {
        recommendations.push(`
            <div class="flex items-start p-3 bg-blue-900 bg-opacity-30 rounded-lg">
                <i class="fas fa-chart-line text-blue-400 mr-3 mt-1"></i>
                <div>
                    <div class="font-semibold">Glucose Monitoring</div>
                    <div class="text-sm text-gray-300">Monitor blood glucose levels regularly as they are in the prediabetic range.</div>
                </div>
            </div>
        `);
    }
    
    // Hypertension recommendations
    if (data.hypertension === 'Yes') {
        recommendations.push(`
            <div class="flex items-start p-3 bg-blue-900 bg-opacity-30 rounded-lg">
                <i class="fas fa-heartbeat text-blue-400 mr-3 mt-1"></i>
                <div>
                    <div class="font-semibold">Blood Pressure Control</div>
                    <div class="text-sm text-gray-300">Continue prescribed medications and monitor blood pressure regularly. Reduce sodium intake.</div>
                </div>
            </div>
        `);
    }
    
    // Smoking recommendations
    if (data.smoking_status === 'smokes') {
        recommendations.push(`
            <div class="flex items-start p-3 bg-blue-900 bg-opacity-30 rounded-lg">
                <i class="fas fa-smoking-ban text-blue-400 mr-3 mt-1"></i>
                <div>
                    <div class="font-semibold">Smoking Cessation</div>
                    <div class="text-sm text-gray-300">Join a smoking cessation program immediately. Consider nicotine replacement therapy.</div>
                </div>
            </div>
        `);
    }
    
    // General recommendations
    recommendations.push(`
        <div class="flex items-start p-3 bg-blue-900 bg-opacity-30 rounded-lg">
            <i class="fas fa-heart text-blue-400 mr-3 mt-1"></i>
            <div>
                <div class="font-semibold">General Health</div>
                <div class="text-sm text-gray-300">Maintain regular physical activity (150 min/week), follow a Mediterranean diet, and get regular health check-ups.</div>
            </div>
        </div>
    `);
    
    recommendationList.innerHTML = recommendations.join('');
}

// Reset assessment
function resetAssessment() {
    // Reset form
    document.getElementById('age').value = 50;
    document.getElementById('ageSlider').value = 50;
    document.getElementById('gender').value = 'Male';
    document.getElementById('ever_married').value = 'No';
    document.querySelector('input[name="hypertension"][value="No"]').checked = true;
    document.querySelector('input[name="heart_disease"][value="No"]').checked = true;
    document.getElementById('avg_glucose_level').value = 120;
    document.getElementById('glucoseSlider').value = 120;
    document.getElementById('bmi').value = 25;
    document.getElementById('bmiSlider').value = 25;
    document.getElementById('work_type').value = 'Private';
    document.getElementById('residence_type').value = 'Urban';
    document.getElementById('smoking_status').value = 'never smoked';
    
    // Reset steps
    currentStep = 1;
    document.getElementById('step1').classList.remove('hidden');
    document.getElementById('step2').classList.add('hidden');
    document.getElementById('step3').classList.add('hidden');
    updateStepIndicators();
    updateNavigationButtons();
    
    // Hide results
    document.getElementById('results').classList.add('hidden');
    
    // Scroll to assessment
    scrollToAssessment();
}

// Download report
function downloadReport() {
    const reportData = {
        timestamp: new Date().toISOString(),
        assessment: assessmentData,
        riskPercentage: document.getElementById('riskPercentage').textContent,
        riskCategory: document.getElementById('riskCategory').textContent
    };
    
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neuropredict-report-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Initialize model comparison chart
function initializeModelChart() {
    const ctx = document.getElementById('modelChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Ensemble', 'XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 'Neural Network'],
            datasets: [{
                label: 'AUC Score',
                data: [0.963, 0.930, 0.925, 0.915, 0.905, 0.895],
                backgroundColor: [
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(147, 51, 234, 0.8)',
                    'rgba(236, 72, 153, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    'rgba(59, 130, 246, 1)',
                    'rgba(147, 51, 234, 1)',
                    'rgba(236, 72, 153, 1)',
                    'rgba(16, 185, 129, 1)',
                    'rgba(245, 158, 11, 1)',
                    'rgba(239, 68, 68, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    ticks: {
                        color: '#9CA3AF'
                    },
                    grid: {
                        color: '#374151'
                    }
                },
                x: {
                    ticks: {
                        color: '#9CA3AF'
                    },
                    grid: {
                        color: '#374151'
                    }
                }
            }
        }
    });
}

// Initialize sliders
function initializeSliders() {
    // Age slider
    const ageSlider = document.getElementById('ageSlider');
    const ageInput = document.getElementById('age');
    ageSlider.addEventListener('input', () => {
        ageInput.value = ageSlider.value;
    });
    ageInput.addEventListener('input', () => {
        ageSlider.value = ageInput.value;
    });
    
    // Glucose slider
    const glucoseSlider = document.getElementById('glucoseSlider');
    const glucoseInput = document.getElementById('avg_glucose_level');
    glucoseSlider.addEventListener('input', () => {
        glucoseInput.value = glucoseSlider.value;
    });
    glucoseInput.addEventListener('input', () => {
        glucoseSlider.value = glucoseInput.value;
    });
    
    // BMI slider
    const bmiSlider = document.getElementById('bmiSlider');
    const bmiInput = document.getElementById('bmi');
    bmiSlider.addEventListener('input', () => {
        bmiInput.value = bmiSlider.value;
    });
    bmiInput.addEventListener('input', () => {
        bmiSlider.value = bmiInput.value;
    });
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeSliders();
    initializeModelChart();
    updateStepIndicators();
    updateNavigationButtons();
    
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add active class to navigation links
    window.addEventListener('scroll', () => {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.nav-link');
        
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (scrollY >= (sectionTop - 200)) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('text-white');
            link.classList.add('text-gray-300');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.remove('text-gray-300');
                link.classList.add('text-white');
            }
        });
    });
});
