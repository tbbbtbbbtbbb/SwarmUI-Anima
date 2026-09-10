/** Use Swarm's model-dependent parameter visibility. */
featureSetChangers.push(() => {
    return currentModelHelper.curCompatClass == 'anima' ? [['swarmanima'], []] : [[], ['swarmanima']];
});
