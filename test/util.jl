
@testset "padding" begin
    @test rpad([1,2,3],5) == [1,2,3,0,0]
    @test rpad([1,2,3],2) == [1,2,3]
    @test rpad([1,2,3],5, 8 ) == [1,2,3,8,8]
    
    @test rpad_to_matrix([[1,2], [3],Int[],[2]])==[
                                                    1 3 0 2
                                                    2 0 0 0
    ]
end;


@testset "names_from" begin
    y_outer = [2,3,4]
    localnames = @names_from begin
        x = 2
        x = 4
        y = y_outer
    end
    
    @test localnames[:x] == 4
    @test localnames[:y] === y_outer
end;


