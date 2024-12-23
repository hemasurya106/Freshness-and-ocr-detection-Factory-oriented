function toggleModel(modelId) {
    const model = document.getElementById(modelId);

    if (model.classList.contains('show')) {
        model.classList.remove('show');
        model.classList.add('hide');
        model.addEventListener('animationend', function hideAfterAnimation() {
            model.classList.remove('hide');
            model.style.display = 'none';
            model.removeEventListener('animationend', hideAfterAnimation);
        });
    } else {
        model.style.display = 'block';
        model.classList.add('show');
    }
}

document.getElementById('task-1-button').addEventListener('click', function () {
    toggleModel('task-1-model');
});

document.getElementById('task-2-button').addEventListener('click', function () {
    toggleModel('task-2-model');
});

document.getElementById('task-3-button').addEventListener('click', function () {
    toggleModel('task-3-model');
});

document.getElementById('task-4-button').addEventListener('click', function () {
    toggleModel('task-4-model');
});
