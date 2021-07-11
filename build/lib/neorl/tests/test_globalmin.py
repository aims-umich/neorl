from neorl.benchmarks.classic import sphere, ackley, cigar

def test_sphere():
    y=sphere([0,0,0,0,0])
    assert y==0

def test_cigar():
    y=cigar([0]*10)
    assert y==0
    
def test_ackley():
    y=ackley([0]*100)
    assert y==0