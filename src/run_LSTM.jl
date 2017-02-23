include("LSTM.jl")

const serv=SwiftService()
#const train_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/train.csv")
const valid_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/dev.csv")

const valid_terms_padded, valid_hsv, encoding = prepare_data(valid_raw)
#const train_terms_padded, train_hsv,  encoding = prepare_data(train_raw, encoding)


const batch_size = 128*4
n_steps=size(valid_terms_padded,1)-1
n_classes = nlabel(encoding)+1
#sess, t =
et = color_to_terms_network(n_classes, n_steps;
        learning_rate = 0.05,
        hidden_layer_size = 32,
        embedding_dim = 16,
        batch_size=batch_size
    )


############################
#obs,pred, oo, pp = run(sess, [Term_obs_onehots, Term_preds_onehots, Term_obs_s_out, Term_preds_s], Dict(X_hsv=>hsv_data, Term_obs_s=>padded_labels))
train_from_terms!(sess, t, train_terms_padded, train_hsv; epochs=3)


cost, acc, perp, preds_o = rough_evalute(sess, t, valid_terms_padded, valid_hsv)

[Pair(a,ind2label(b,encoding)) for (a,b) in sort(reverse.(collect(countmap(train_terms_padded[2,:]))), rev=true)[2:end]]

collect(enumerate(ind2label.(2:50, encoding)))

unique_cols = first.(unique(last, enumerate(eachobs(preds_o))))
pls = ind2label.(Int.(preds_o[:,unique_cols]), encoding)'

ols_coded=valid_terms_padded
ols_coded[ols_coded.==0]=1
ols = ind2label.(Int.(ols_coded[2:end, unique_cols]), encoding)'

join(mapslices(x->join(x," ") , [ols fill("->", size(unique_cols,1)) pls], 2), "\n") |> print

methods(ind2label)



@testset "masks" begin
  mask_sess = Session(Graph())
  a_val = (
  [2.  5  8  9  2
   3.  6  4  4  9
   4.  7  3  2  4
   1.  4  2  4  4]
  )
  A=constant(a_val)
  M = constant([1, 0, 2, 4])
  @show masked = run(mask_sess, squeeze(find(get_mask(M)), [2]))
  @show masked = run(mask_sess, gather_nd(A, [1 3 4]'))
  @show masked = run(mask_sess, gather_nd(A, find(get_mask(M))))


  masked = run(mask_sess, apply_mask(A, get_mask(M)))

  @test all(masked[1,:] .== a_val[1,:])
  @test all(masked[2:3,:] .== a_val[3:4,:])
end;

##MASK test code
#TODO
if test_it
      @assert isa(run(sess, cost, Dict(X_hsv=>hsv_data, Term_obs_s=>padded_labels)), Number )
      ## MASK TEST######################
      ll, ll2 = run(sess, [ LL_masked,
          reshape(tile(expand_dims(get_mask(Term_obs_s_out),2),[1,1,8]).*concat(0, expand_dims.(Ls, Scalar(0))), [batch_size*n_steps, n_classes])
      ], Dict(X_hsv=>hsv_data, Term_obs_s=>padded_labels))
      @assert ll ≈ ll2
  end

sess = Session(Graph())
x = placeholder(Int64)
run(sess, get(cast(x, Int32)), Dict(x=>2))
