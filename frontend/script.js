// Global variables
let chart = null;

// Elastic Beanstalk backend URL
const API_URL = 'http://localhost:8080';

// Function to fetch prediction from the backend
function fetchPrediction(data) {
    fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Prediction data:', data);
        updateUI(data);
    })
    .catch(error => {
        console.error('Error:', error);
        displayError('An error occurred while fetching the prediction. Please try again.');
    });
}

// Function to update the UI with prediction results
function updateUI(data) {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('results').style.display = 'block';

    document.getElementById('average-price').textContent = `$${data.average.toFixed(2)}`;
    document.getElementById('min-price').textContent = `$${data.min.toFixed(2)}`;
    document.getElementById('max-price').textContent = `$${data.max.toFixed(2)}`;
    document.getElementById('accuracy').textContent = `±$${data.accuracy.toFixed(2)}`;

    createOrUpdateChart(data.dates, data.prices);
    displayBestDays(data.best_days);
}

// Function to create or update the chart
function createOrUpdateChart(dates, prices) {
    const ctx = document.getElementById('price-chart').getContext('2d');

    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Predicted Price',
                data: prices.map((price, index) => ({x: dates[index], y: price})),
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'MMM d'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Price ($)'
                    }
                }
            }
        }
    });
}

// Function to display best days to book
function displayBestDays(bestDays) {
    const bestDaysList = document.getElementById('best-days-list');
    bestDaysList.innerHTML = '';
    bestDays.forEach(day => {
        const li = document.createElement('li');
        li.textContent = `${day.date}: $${day.price.toFixed(2)}`;
        bestDaysList.appendChild(li);
    });
}

// Function to display error messages
function displayError(message) {
    const errorElement = document.getElementById('error-message');
    errorElement.textContent = message;
    errorElement.style.display = 'block';
    document.getElementById('loading').style.display = 'none';
}

// Event listener for form submission
document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();
    document.getElementById('error-message').style.display = 'none';
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    const origin = document.getElementById('origin').value.toUpperCase();
    const destination = document.getElementById('destination').value.toUpperCase();
    const date = document.getElementById('date').value;

    if (!origin || !destination || !date) {
        displayError('Please fill in all fields.');
        return;
    }

    fetchPrediction({origin, destination, date});
});

// Initialize date picker with a minimum date of today
document.addEventListener('DOMContentLoaded', function() {
    const dateInput = document.getElementById('date');
    const today = new Date().toISOString().split('T')[0];
    dateInput.min = today;
});
