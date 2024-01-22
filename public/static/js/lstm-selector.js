document.addEventListener('DOMContentLoaded', function () {
  const yearBtn = document.getElementById('year-btn');
  const monthBtn = document.getElementById('month-btn');
  const dayBtn = document.getElementById('day-btn');
  const generateBtn = document.getElementById('generate-btn');

  const yearDropdown = document.getElementById('year-dropdown');
  const monthDropdown = document.getElementById('month-dropdown');
  const dayDropdown = document.getElementById('day-dropdown');

  function toggleDropdown(dropdown) {
    dropdown.style.display = dropdown.style.display === 'block' ? 'none' : 'block';
  }

  function updateButtonText(button, selectedOption) {
    button.textContent = selectedOption.textContent;
  }

  function createOptionClickHandler(button, dropdown) {
    return function (event) {
      const selectedOption = event.target;
      dropdown.style.display = 'none';
      updateButtonText(button, selectedOption);
    };
  }

  yearBtn.addEventListener('click', function () {
    toggleDropdown(yearDropdown);
    monthDropdown.style.display = 'none';
    dayDropdown.style.display = 'none';
  });

  monthBtn.addEventListener('click', function () {
    toggleDropdown(monthDropdown);
    yearDropdown.style.display = 'none';
    dayDropdown.style.display = 'none';
  });

  dayBtn.addEventListener('click', function () {
    toggleDropdown(dayDropdown);
    yearDropdown.style.display = 'none';
    monthDropdown.style.display = 'none';
  });

  generateBtn.addEventListener('click', function () {
    const selectedYear = yearBtn.textContent;
    const selectedMonth = monthBtn.textContent;
    const selectedDay = dayBtn.textContent;

    console.log(`Selected Date: ${selectedYear}-${selectedMonth}-${selectedDay}`);
    window.location.href = `/generate-prediction-lstm?year=${selectedYear}&month=${selectedMonth}&day=${selectedDay}`;
  });

  // Generate year options (e.g., from 2020 to 2030)
  generateOptions(2020, 2030, yearDropdown);

  // Generate month options (e.g., from January to December)
  generateOptions(1, 12, monthDropdown);

  // Generate day options (e.g., from 1 to 31)
  generateOptions(1, 31, dayDropdown);

  function generateOptions(start, end, container) {
    for (let i = start; i <= end; i++) {
      const option = document.createElement('button');
      option.textContent = i;
      option.addEventListener('click', createOptionClickHandler(container.previousElementSibling, container));
      container.appendChild(option);
    }
  }
});
