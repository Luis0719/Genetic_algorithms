from random import randrange


def get_pivots(top_limit, invalid_pivot):
    # Get 2 random numbers which will work as pivots
    pivot1 = randrange(top_limit)
    pivot2 = randrange(top_limit)
    
    # Validate that the pivots are valid based on the validation function
    while invalid_pivot(pivot1, pivot2):
        pivot2 = randrange(top_limit)

    # Make sure pivot1 is lower than pivot2
    if pivot1 > pivot2:
        tmp = pivot1
        pivot1 = pivot2
        pivot2 = tmp

    return pivot1, pivot2

def operator1(cromosome):
    '''
        Operator 1 will get 2 pivots and will reverse the order of the items between this 2 pivots
        example:
            item = [1,2,3,4,5,6,7,8]
            pivot1 = 2
            pivot2 = 5
            result = [1,2,5,4,3,6,7,8]
    '''
    
    # Define a function which will tell when the selection of pivots is invalid
    def invalid_pivot(pivot1, pivot2):
        # Make sure there's at least a difference of 2 between the pivots and that at least 1 gen doesn't change its position (NOTE: Last 'gen' doesn't count because it's the value of the aptitud function)
        return (pivot2 < pivot1+2 and pivot2 > pivot1-2) or (abs(pivot2-pivot1) == (cromosome_size-1))

    cromosome_size = len(cromosome)+1
    pivot1, pivot2 = get_pivots(cromosome_size, invalid_pivot)

    cromosome = cromosome[:pivot1] + cromosome[pivot1:pivot2][::-1] + cromosome[pivot2:]
    return cromosome

def operator2(cromosome):
    '''
        Operator 2 will get 2 'blocks' of gens of equal size and will switch its places
        example:
        item = [1,2,3,4,5,6,7,8]
        block1 = [2,3]
        block2 = [6,7]
        result = [1,6,7,4,5,2,3,8]
    '''

    # Define a function which will tell when the selection of pivots is invalid
    def invalid_pivot(pivot1, pivot2):
        # Since each block must have at least 1 element, the difference between the pivots must be equal or bigger than 2
        return abs(pivot2 - pivot1) < 2

    cromosome_size = len(cromosome)+1
    pivot1, pivot2 = get_pivots(cromosome_size, invalid_pivot)
    
    # For this operator, pivot1 will be the starting position of the block and pivot2 will be then last position of block2
    # Now we need to calculate the size of the block.
    block_size = randrange(int((pivot2-pivot1)/2))+1
    block1 = cromosome[pivot1:pivot1+block_size]
    block2 = cromosome[pivot2-block_size:pivot2]

    # Build the new cromosome with the blocks switched
    cromosome = cromosome[:pivot1] + block2 + cromosome[pivot1+block_size:pivot2-block_size] + block1 + cromosome[pivot2:]
    return cromosome

def breeder_factory(option, cromosome):
    if option == 0:
        return operator1(cromosome)
    
    if option == 1:
        return operator2(cromosome)

    print("Invalid breeder option {}".format(str(option)))
    return cromosome
