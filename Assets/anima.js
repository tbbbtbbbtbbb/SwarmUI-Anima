/** Use Swarm's model-dependent parameter visibility. */
featureSetChangers.push((features, removals) => {
    if (currentModelHelper.curCompatClass != 'anima') {
        removals.push('swarmanima');
    }
});
