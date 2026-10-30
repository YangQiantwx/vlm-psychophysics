const video = document.querySelector('#recorded-video');
const choice = document.querySelector('#recorded-choice');
const status = document.querySelector('#recorded-status');
choice.addEventListener('change', () => {
  video.pause();
  status.textContent = '';
  video.src = choice.value;
  video.poster = choice.selectedOptions[0].dataset.poster || '';
  video.load();
});
video.addEventListener('error', () => {
  status.textContent = 'This preview could not be loaded. Please reload or choose another example.';
});
video.addEventListener('loadeddata', () => { status.textContent = ''; });
