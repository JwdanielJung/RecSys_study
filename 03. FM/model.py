import pandas as pd
import numpy as np
import torch 
import torch.nn.functional as F

class FeaturesLinear(torch.nn.Module):
    
    def __init__(self, field_dims, output_dim=1):
        # torch.nn.Module 상속받았다는 의미로 super() 선언
        super().__init__()
        # fc = fully connected 의미
        # layer를 지나면 embedding 값이 도출됨을 의미
        # 선형 변환을 통해 look-up table에서 필요한 임베딩 값 return 해줌 (주로 범주형 -> 연속형 시 사용)
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        # cumsum: cumulate sum / * = array unpacked
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int_)

    def forward(self, x):
        # tensor x size = (batch_size, num_field)

        # tensor.new_tensor(x) -> x라는 tensor 값 copy 하는 method
        # 이때 new_tensor 내 x의 데이터 타입은 tensor와 동일하게됨!! -> 타입 맞춰줄 때 사용
        # unsqueeze(x) -> x 차원 삭제 
        # 0 ~ field num까지 모든 값을 array로 만든 것이 offsets! -> 그 값 더해주면 각각 index 부여가능
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        
        # self.fc(x)가 결국 x에 대한 선형변환! -> wixi 의미 => return 값: sum(wi*xi) + w0 의미
        return torch.sum(self.fc(x), dim=1) + self.bias


# factorization 통해 얻은 vi 구현
class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int_)
        # 이전 이후 node에 의존적인 가중치 초기화 방법
        # sigmoid에 대해 효과 good / ReLU에선 0 수렴 -> 다른 초기화 방법 사용
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)
    
class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum = True):
        super().__init__()
        self.reduce_sum = reduce_sum # 왜쓰징 ?.?

    def forward(self, x):
        # x: float tensor (batch_size, num_fields, embed_dim)
        # linear complexity로 변환한 식 기반으로 forward 함수 구성

        square_of_sum = torch.sum(x, dim = 1) ** 2 # 제곱한 값들 field 별로 합
        sum_of_square = torch.sum(x **2, dim = 1) # field 별로 제곱 합

        diff = square_of_sum - sum_of_square

        if self.reduce_sum:
            diff = torch.sum(diff, dim = 1, keepdim=True)

        return 0.5 * diff

class FactorizationMachineModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))
    


    


    