using LIBLINEAR

# computation validation
iris = readcsv(joinpath(dirname(@__FILE__), "iris.csv"))
instances = convert(Matrix{Float64}, iris[:, 1:4]')

labels = iris[:, 5]
y = Float64[i ? 1 : -1 for i = labels .== "virginica"]

P = sortperm(y, rev = true)
A = instances
w = ones(size(y))

y = y[P]
A = A[:,P]
model = linear_train(y, A, w; verbose=true, solver_type=Cint(3), eps = 1e-11)

gc()

v = sparsevec(model.SVI + 1, model.SV, size(y,1))

pre(A) = A'*(A*(y.*v))

SV   = v .!= 0
z    = pre(A)
α    = vecdot(z,y.*v)
pval = vecdot(w, max( 1 - y.*z, 0)) + 0.5*α
dval = 0.5*α - sum(v)

println(pval + dval)

#(class, decvalues) = linear_predict(model, instances[:, 2:2:end], verbose=true)
#correct = Bool[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
#@assert (class .== labels[2:2:end]) == correct

