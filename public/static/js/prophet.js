const startSlider = document.getElementById("start-date");
const endSlider = document.getElementById("end-date");
const num_days = document.getElementById("num_days");
const startValue = document.getElementById("start-value");
const endValue = document.getElementById("end-value");

// Start: 21.01.2022
// End: 15.11.2023
startDateLimit = "2022-01-21"
endDateLimit = "2023-11-15"

const startDateRange = new Date(startDateLimit).getTime();
const endDateRange = new Date(endDateLimit).getTime();

// Set the slider's min and max values to correspond to your date rang
startSlider.min = startDateRange;
startSlider.max = endDateRange;
endSlider.min = startDateRange;
endSlider.max = endDateRange;

// Initialize the displayed values with the default slider values
startValue.textContent = new Date(parseInt(startSlider.value)).toLocaleDateString();
endValue.textContent = new Date(parseInt(endSlider.value)).toLocaleDateString();

// Add event listeners to update the displayed values when sliders are adjusted
startSlider.addEventListener("input", () => {
    startValue.textContent = new Date(parseInt(startSlider.value)).toLocaleDateString();
});

endSlider.addEventListener("input", () => {
    endValue.textContent = new Date(parseInt(endSlider.value)).toLocaleDateString();
});

const generateButton = document.getElementById("generate");
generateButton.addEventListener("click", () => {
    // Calculate the selected start and end dates
    const selectedStartDate = new Date(parseInt(startSlider.value)).toLocaleDateString();
    const selectedEndDate = new Date(parseInt(endSlider.value)).toLocaleDateString();
    const numberOfDays = parseInt(num_days.value)

    console.log(`Selected Start Date: ${selectedStartDate}`);
    console.log(`Selected End Date: ${selectedEndDate}`);

    const preloader = document.getElementById('preloader');
    const loaderImage = preloader.querySelector('.loader');

    preloader.style.display = 'flex';
    loaderImage.style.display = 'block';

    window.location.href = `/generate-graph?start=${selectedStartDate}&end=${selectedEndDate}&num_days=${numberOfDays}`;
});