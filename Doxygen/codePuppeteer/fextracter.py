import json

from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from typing import Tuple, List, Dict, Optional
import regex


class InvalidFormat(Exception):
    pass


class EnhancementFailure(Exception):
    pass


class ParamDecl:
    def __init__(self, sdecl: str, name_loc: Tuple[int, int], default_value: str, is_ref: bool = False):
        self.sdecl: str = sdecl  # `type name`
        self.name_loc: Tuple[int, int] = name_loc
        self.alias: Optional[str] = None
        self.default_value: str = default_value
        self.given_value: Optional[str] = None
        self.is_ref = is_ref  # 转指针已经在外部搞定，这里只需要取传入变量的地址

    def assign(self, given_value: str):
        self.given_value = given_value

    def buffer_name(self, alias: str):
        self.alias = alias

    def reset(self):
        self.given_value = None
        self.alias = None

    def __call__(self):
        if self.alias is None:
            temp = self.sdecl + '='
            if self.is_ref:
                temp += '&'
            temp += self.given_value if self.given_value is not None else self.default_value
            temp += ';'
            return temp
        else:
            temp = self.sdecl[:self.name_loc[0]] + self.alias + self.sdecl[self.name_loc[1]:] + '='
            if self.is_ref:
                temp += '&'
            if self.given_value is not None:
                temp += self.given_value
            else:
                temp += self.default_value
            temp += f";{self.sdecl}={self.alias};"
            return temp

    def __repr__(self):
        return f"「decl: {self.sdecl} ,default_value={self.default_value}, given_value={self.given_value}, " \
               f"alias={self.alias}, is_ref={self.is_ref}」"


class ParamDecl2:
    def __init__(self, type, name, array, defval):
        self.type = type + '*' if array else type
        self.name = name
        self.defval = defval
        self.given_value = None
        self.alias = None
        self.is_ref = '&' in self.type
        self.type = self.type.replace('&', '*')

    def assign(self, given_value: str):
        self.given_value = given_value

    def buffer_name(self, alias: str):
        self.alias = alias

    def reset(self):
        self.given_value = None
        self.alias = None

    def __call__(self):
        given_value = self.given_value if self.given_value is not None else self.defval
        if self.is_ref:
            given_value = f"&{given_value}"
        if self.alias is not None:
            return f"{self.type} {self.alias} = {given_value};{self.type} {self.name} = {self.alias};"
        else:
            return f"{self.type} {self.name} = {given_value};"

    def __repr__(self):
        return f"「2decl: {self.type} {self.name} ,default_value={self.defval}, given_value={self.given_value}, " \
               f"alias={self.alias}, is_ref={self.is_ref}」"


# 不考虑: int var = a++ + foo(a);
# 不考虑 define ，此时行末可能没有 `;`
def structural_normalize(tokens) -> list:
    """
    所有单行 block 表达式加上 {}
    递归、不断往后找一句完整的句子
    """

    # cache = []
    def __hierarchy_pack(istart) -> Tuple[int, int]:
        # 0: undetermined, 1: `;`, 2: find keyword if/for/while/switch/[else], to find first `(`,
        # 3: find `()` pair to make cnt == 0 [else automatically satisfy], 4: find `{`, 5: find `}`
        # print("enter new!")
        # cache.append(istart)
        status = 0
        cnt = 0
        iloc = istart
        while True:
            if iloc >= len(tokens):
                if status == 0:
                    # print("end!")
                    # cache.remove(istart)
                    return 0, iloc
                else:
                    # print(status)
                    raise EnhancementFailure()
            if status == 0:
                if tokens[iloc][0] == Token.Keyword and tokens[iloc][1] in ['if', 'for', 'while', 'switch']:
                    status = 2
                elif tokens[iloc] == [Token.Keyword, 'else']:
                    status = 3
                elif tokens[iloc] == [Token.Punctuation, '{']:
                    status = 4
                elif tokens[iloc] == [Token.Punctuation, '}']:
                    # print(f"token = {tokens[iloc]}, status = {status}")
                    # print("return!")
                    # cache.remove(istart)
                    # print("get:", ''.join(map(lambda x: x[1], tokens[istart:iloc+1])))
                    return 5, iloc + 1
                elif tokens[iloc][0] != Token.Text.Whitespace and tokens[iloc][0] not in Token.Comment:
                    status = 1
                    continue
            elif status == 1:
                # do-while 结构
                if tokens[iloc] == [Token.Keyword, 'do']:
                    res, iend = __hierarchy_pack(iloc + 1)
                    if res == 1:
                        tokens.insert(iend, [Token.Punctuation, '}'])
                        tokens.insert(iloc + 1, [Token.Punctuation, '{'])
                        iloc = iend + 2
                    elif res == 4:
                        iloc = iend
                    else:
                        raise EnhancementFailure()
                    # print(f"token = {tokens[iloc]}, status = {status}")
                    continue
                elif tokens[iloc] == [Token.Punctuation, ';']:
                    # print(f"token = {tokens[iloc]}, status = {status}")
                    # print("return!")
                    # cache.remove(istart)
                    # print("get:", ''.join(map(lambda x: x[1], tokens[istart:iloc+1])))
                    return status, iloc + 1
            elif status == 2:
                if tokens[iloc] == [Token.Punctuation, '(']:
                    status = 3
                    cnt = 1
                elif tokens[iloc][0] != Token.Text.Whitespace and tokens[iloc][0] not in Token.Comment:
                    raise EnhancementFailure()
            elif status == 3:
                if cnt != 0:
                    if tokens[iloc] == [Token.Punctuation, '(']:
                        cnt += 1
                    elif tokens[iloc] == [Token.Punctuation, ')']:
                        cnt -= 1
                else:
                    res, iend = __hierarchy_pack(iloc)
                    if res == 1:
                        # 颠倒顺序会导致错误
                        tokens.insert(iend, [Token.Punctuation, '}'])
                        tokens.insert(iloc, [Token.Punctuation, '{'])
                        iloc = iend + 2
                    elif res == 4:
                        iloc = iend
                    else:
                        raise EnhancementFailure()
                    # print(f"token = {tokens[iloc]}, status = {status}")
                    # print("return!")
                    # cache.remove(istart)
                    # print("get:", ''.join(map(lambda x: x[1], tokens[istart:iloc])))
                    return 1, iloc
            elif status == 4:
                res, iend = __hierarchy_pack(iloc)
                iloc = iend
                if res == 5:
                    # print(f"token = {tokens[iloc] if iloc < len(tokens) else None}, status = {status}")
                    # print("return!")
                    # cache.remove(istart)
                    # print("get:", ''.join(map(lambda x: x[1], tokens[istart:iloc])))
                    return status, iend
                else:
                    # print(f"token = {tokens[iloc]}, status = {status}")
                    continue
            # print(f"token = {tokens[iloc]}, status = {status}")
            iloc += 1

    __hierarchy_pack(0)
    return tokens


# TODO: 暂时不处理同名函数
class FunctionDecl:
    class ReturnPoint:
        def __repr__(self):
            return str(__class__.__name__)

    def __init__(self, rtype: str, fname: str, params: Dict[str, ParamDecl], body: list, loc_in_file: Tuple[int, int],
                 line_number_before_body: int, roi: List[int] = None):
        self.rtype = rtype
        self.rtype_is_void = "void" in self.rtype
        self.fname = fname
        self.params_keys = list(params.keys())
        self.params = list(params.values())
        self.body = list(map(
            lambda t: [[Token.Punctuation, '('], [Token.Operator, '*'], [t[0], t[1]], [Token.Punctuation, ')']] if
            t[0] == Token.Name and t[1] in params.keys() and params[t[1]].is_ref else [[t[0], t[1]]], body))
        self.body = sum(self.body, [])  # 不包含 {}
        flag = False
        if self.rtype_is_void:
            for s in filter(lambda temp: temp == [Token.Keyword, 'return'], self.body):
                s[1] = 'break'
        else:
            cnt = 0
            for s in self.body:
                if flag and s == [Token.Punctuation, ';']:
                    self.body.insert(cnt, [Token.Punctuation, ')'])
                    self.body.insert(cnt + 2, [Token.Keyword, 'break'])
                    self.body.insert(cnt + 3, [Token.Punctuation, ';'])
                    flag = False
                elif s == [Token.Keyword, 'return']:
                    s[0] = FunctionDecl.ReturnPoint()
                    self.body.insert(cnt + 1, [Token.Operator, '='])
                    self.body.insert(cnt + 2, [Token.Punctuation, '('])
                    flag = True
                cnt += 1
        self.loc_in_file = loc_in_file
        self.line_number_before_body = line_number_before_body
        # 融合是不需要 body 前的部分
        self.roi = roi if roi is not None else []  # 从 body 开始算的相对位置

        self.body_length = ''.join(map(lambda x: x[1], self.body)).count('\n')

        self.param_idx = 0
        self.TEMPi = 0

    def pass_roi(self, roi: List[int]):
        """
        传入相对于整个函数开始的 roi 区域
        :param roi: List[int]
        :return:
        """
        self.roi = list(map(lambda x: x - self.line_number_before_body, roi))

    def transfer_param(self, value: str):
        """
        :param value: 传入参数
        :return:
        """
        # print("value:", value)
        p = self.params[self.param_idx]
        p.assign(value)
        if value in self.params_keys:
            p.buffer_name(f'TEMP{self.TEMPi}')
            self.TEMPi += 1
        self.param_idx += 1

    def reset(self):
        self.param_idx = 0
        self.TEMPi = 0
        for p in self.params:
            p.reset()

    def __call__(self, return_value: str) -> Tuple[str, str]:
        """
        生成调用展开后的代码片段
        :param return_value: 承接结算结果的返回变量
        """
        body = ''.join(map(lambda x: return_value if isinstance(x[0], FunctionDecl.ReturnPoint) else x[1], self.body))
        # print("self.params", self.params)
        return (f"{self.rtype} {return_value};" if not self.rtype_is_void else ';',
                r"do {" +
                f"{' '.join(map(lambda x: x(), self.params))}" +
                f"{body}" +
                r"} while(0);")

    def str_body(self):
        return ''.join(map(lambda x: x[1], self.body))

    def __repr__(self):
        return str({
            "returnType": self.rtype,
            "funcName": self.fname,
            "params": self.params,
            "funcBody": ''.join(map(lambda x: x[1], self.body))
        })


class FunctionDeclOrig:
    def __init__(self, tokens: list, fname: str, body_iloc: int, loc_in_file: Tuple[int, int],
                 line_number_in_file: int, line_number_before_body: int, roi: List[int] = None):
        self.fname = fname
        self.tokens = tokens
        self.body_iloc = body_iloc  # { 处
        self.line_number_in_file = line_number_in_file
        self.iReturn = 0

        self.loc_in_file = loc_in_file
        self.roi = roi if roi is not None else []
        # line_number_before_body = ''.join(map(lambda x: x[1], self.tokens[:body_iloc])).count('\n')
        self.line_number_before_body = line_number_before_body
        self.roi = list(map(lambda x: x + self.line_number_before_body, self.roi))

    def get_orig(self):
        return ''.join(map(lambda x: x[1], self.tokens))

    def pass_roi(self, roi: List[int]):
        """
        传入相对于整个函数开始的 roi 区域，orig 不需要修改传入的 roi
        :param roi: List[int]
        :return:
        """
        self.roi = roi

    def expand(self, callee: FunctionDecl):
        """
        # TODO: 通过函数指针的调用暂不支持
        (1) 在 caller 最开始声明返回变量
        (2) 在 call site 前合适点插入 callee 改写后的函数体
        (3) 在 call site 修改为返回变量
        :param callee:
        :return:
        """
        callee.reset()
        is_exist = False
        # 加速处理：不存在这个变量就直接退出
        if len(list(filter(lambda t: t == [Token.Name, callee.fname], self.tokens))) == 0:
            print(f"Function {callee.fname} not found in {self.fname}")
            return False, None, None
        # 静态函数指针支持
        flag = False
        callee_fname = callee.fname
        code = self.get_orig()
        for match in regex.finditer(fr"\w+\s*\(\*(?P<ptr>[a-zA-Z_][a-zA-Z0-9_]*)\)\s*\([^)]*\)\s*=\s*{callee_fname}\s*;", code):
            callee_fname = match.group("ptr")
            flag = True
            break
        if not flag:
            # 可能定义后再赋值
            for match in regex.finditer(fr"(?P<ptr>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*{callee_fname}\s*;", code):
                callee_fname = match.group("ptr")
                break
                # TODO: 多个名字

        # 延迟写主要用于解决列表插入和修改导致索引变化，影响 for 遍历
        # (i, str) - Insert str at pos i, (i, j, str) - Replace the contents of pos i to pos j with str
        delayed_write: List[List[int, str] or List[int, int, str]] = []
        # 寻找函数调用
        returnParamDecl = ""
        status = 0  # 0: find var name; 1: confirm usage (followed by `(`); 2: find the end of call
        insert_pos = 0
        cnt = 0
        first_iloc = -1
        last_iloc = -1
        # delimiter ==`;` | ==`}` | in `Token.Comment`
        # print("caller:", self.fname)
        # print("callee:", callee.get_orig())
        # print("callee decl:", callee.params)
        for iloc in range(self.body_iloc, len(self.tokens)):
            # 寻找变量名
            if status == 0 and self.tokens[iloc] == [Token.Name, callee_fname]:
                status = 1
                first_iloc = iloc
                continue
            # 确认为函数调用，即 followed by (
            elif status == 1:
                if self.tokens[iloc][0] in [Token.Text.Whitespace, Token.Comment.Multiline]:
                    continue
                elif self.tokens[iloc] == [Token.Punctuation, '(']:
                    status = 2
                    last_iloc = iloc + 1
                    cnt += 1
                    continue
                else:
                    status = 0
                    continue
            elif status == 2:
                if self.tokens[iloc] == [Token.Punctuation, '(']:
                    cnt += 1
                elif self.tokens[iloc] == [Token.Punctuation, ')']:
                    cnt -= 1
                if cnt == 0 or (cnt == 1 and self.tokens[iloc][1] == ','):
                    if last_iloc != iloc:
                        # print(''.join(map(lambda x: x[1], self.tokens[last_iloc: iloc])))
                        callee.transfer_param(''.join(map(lambda x: x[1], self.tokens[last_iloc: iloc])))
                    last_iloc = iloc + 1
                if cnt == 0:
                    status = 0
                    # insert
                    returnVar = ""
                    if not callee.rtype_is_void:
                        returnVar = f"RETURN{self.iReturn}"
                        self.iReturn += 1
                    return_param_decl, rewrite_body = callee(returnVar)
                    delayed_write.append([insert_pos, rewrite_body])
                    delayed_write.append([first_iloc, iloc + 1, returnVar])
                    returnParamDecl += return_param_decl
                    callee.reset()
                    is_exist = True
                continue
            else:
                # 其他情况时寻找适合插入函数体的位置
                if (self.tokens[iloc][0] == Token.Punctuation and self.tokens[iloc][1] in [';', '{', '}']) or \
                        self.tokens[iloc][0] in Token.Comment:
                    insert_pos = iloc + 1
                continue
        results = list(map(lambda x: x[1], self.tokens))
        delayed_write.sort(key=lambda x: x[0])
        roi = self.roi.copy()
        # print("roi=", roi)
        for i in range(len(delayed_write)):
            item = delayed_write[i]
            if len(item) == 2:
                results.insert(*item)
                for j in range(i + 1, len(delayed_write)):
                    if len(delayed_write[j]) == 2:
                        delayed_write[j][0] += 1
                    else:
                        delayed_write[j][0] += 1
                        delayed_write[j][1] += 1
                line_before = ''.join(results[:item[0]]).count('\n')
                # print("line_before=", line_before)
                # 假设插入都发生在行尾，尽管这并不一定成立
                roi = list(map(lambda x: x + callee.body_length if x > line_before else x, roi)) + \
                      list(map(lambda x: x + line_before, callee.roi))
                # print("roi=", roi)
            else:  # 3
                del results[item[0]: item[1]]
                results.insert(item[0], item[2])
                delta = item[1] - item[0] - 1
                for j in range(i + 1, len(delayed_write)):
                    if len(delayed_write[j]) == 2:
                        delayed_write[j][0] -= delta
                    else:
                        delayed_write[j][0] -= delta
                        delayed_write[j][1] -= delta
        if len(returnParamDecl) != 0:
            results.insert(self.body_iloc + 1, returnParamDecl)
        return is_exist, ''.join(results) if is_exist else None, roi


class FExtracter:
    def __init__(self, code: str, lan: str):
        assert lan in ['Objective-C', 'C++'], f"Sorry, FExtracter only support C/C++ code for now! given: {lan}"
        self.__code = code
        self.__lan = lan
        self.__lexer = get_lexer_by_name(lan)
        self.__tokens_unprocessed = self.__lexer.get_tokens_unprocessed(code)
        # Split code to prevent lexical parsing errors from propagating, TODO: {} in string
        # slocs = [0]
        # for res in regex.finditer(r"(\{((?>[^\{\}]+|(?1))*)\})", code):
        #     slocs.append(res.end())
        # if slocs[-1] != len(code):
        #     slocs.append(len(code))
        # for i in range(len(slocs) - 1):
        #     self.__tokens_unprocessed.extend(
        #         list(map(lambda x: (x[0] + slocs[i], x[1], x[2]),
        #                  self.__lexer.get_tokens_unprocessed(code[slocs[i]: slocs[i + 1]]))))
        # [iloc, sloc, type, string] where iloc - index in list, sloc - index in string
        self.__tokens_unprocessed = list(map(lambda r: (r[0], *r[1]), enumerate(self.__tokens_unprocessed)))
        # cache functions that have been processed
        self.__cache = {}  # function name: str -> function definition: FunctionDecl

        self.__ref_params: List[str] = []

    def get_lan(self):
        return self.__lan

    def __get_function_return_type(self, iloc: int):
        """
        获得函数定义的起始边界
        :param iloc: 函数名在 self.__tokens_unprocessed 中的索引
        :return:
        """
        cur_iloc = iloc - 1
        while cur_iloc >= 0:
            curToken = self.__tokens_unprocessed[cur_iloc]
            if curToken[2] in Token.Comment or \
                    (curToken[2] in Token.Punctuation and curToken[3] in [';', '}']):
                return self.__tokens_unprocessed[cur_iloc + 1][0], self.__tokens_unprocessed[iloc][0]
            cur_iloc -= 1
        return 0, self.__tokens_unprocessed[iloc][0]

    def __get_function_param_decl(self, iloc) -> Tuple[int, int]:
        """
        返回函数参数定义部分 (...)
        核心思想：与语法分析器相同的栈思想 () 匹配
        :param iloc:
        :return: (iloc_start, iloc_end, iloc_end) - iloc_end 用于后续 body 查询
        """
        round_brackets = filter(lambda t: t[0] >= iloc and t[2] == Token.Punctuation and (t[3] == '(' or t[3] == ')'),
                                self.__tokens_unprocessed)
        count: int = 0
        start_iloc: int = -1
        end_iloc: int = -1
        for round_bracket in round_brackets:
            if round_bracket[3] == '(':
                if count == 0:
                    start_iloc = round_bracket[0]
                count += 1
            else:
                end_iloc = round_bracket[0]
                count -= 1
            if count == 0:
                break
        assert start_iloc != -1 and end_iloc != -1, "[ERROR] bracket failure!"
        # 后面一定有函数体，所以 end_iloc 直接 +1 取后一项
        return start_iloc, end_iloc + 1

    def __parse_param_decl(self, code):
        """TODO: 引用传递，函数指针
        能处理以下几种情况：
        (1) type varName 包括引用传递 &
        (2) type varName = value 包括引用传递 &
        (3) type varName[]

        用分词而非正则表达式的好处是可以避免 comment 和 string 的影响
        :param start_iloc: 参数开始位置，不包括 ()
        :param end_iloc: 参数结束位置（开闭规则：左闭右开）
        :return: name, {"name": str, "decl": str, "default_value": str or None, "given_value": str or None}
        """
        # print("single: ", self.__code[self.__tokens_unprocessed[start_iloc][1]: self.__tokens_unprocessed[end_iloc][1]])
        '''
        # 分隔符号
        rights = ['}', '>', ']', ')']
        lefts = ['{', '<', '[', '(']

        # Step I: 找等号，跳过可能的没有意义的`=`
        signpost1 = end_iloc  # 记录后续操作的终点
        count = 0
        default_value = None
        for iloc in reversed(range(start_iloc, end_iloc)):
            if self.__tokens_unprocessed[iloc][3] in rights:
                count += 1
            elif self.__tokens_unprocessed[iloc][3] in lefts:
                count -= 1
            elif self.__tokens_unprocessed[iloc][3] == '=' and count == 0:
                signpost1 = iloc
                default_value = self.__code[
                                self.__tokens_unprocessed[signpost1 + 1][1]: self.__tokens_unprocessed[end_iloc][1]
                                ].strip()
                break

        # Step II: 特殊情况
        # special case 1: 函数指针 type (*fn)(param_types)
        PCRE_fpointer = r".*?\(\*\s*(?P<name>\w+)\s*\)\s*(\(((?>[^\(\)]+|(?2))*)\))\s*"
        _code = self.__code[self.__tokens_unprocessed[start_iloc][1]: self.__tokens_unprocessed[signpost1][1]]
        res = regex.match(PCRE_fpointer, _code)
        if res is not None:
            name = res.group("name")
            assert _code == res.group(), f"PCRE_fpointer is not complete.\n_code: {_code}\nres.group(): {res.group()}"
            return name, ParamDecl(_code + '=', res.span(), default_value)

        # special case 2: 数组 type name[] -> type* name
        # 跳过没有意义的 `[]`
        signpost2 = signpost1
        count = 0
        _u = -1
        _b = -1
        for iloc in reversed(range(start_iloc, signpost2)):
            if self.__tokens_unprocessed[iloc][3] in rights:
                if self.__tokens_unprocessed[iloc][3] == ']' and count == 0:
                    _u = iloc + 1  # 左开右闭
                count += 1
            elif self.__tokens_unprocessed[iloc][3] in lefts:
                count -= 1
                if self.__tokens_unprocessed[iloc][3] == '[' and count == 0:
                    _b = iloc
                    signpost2 = iloc
                    break
        assert _u == -1 and _b == -1 or _u != -1 and _b != -1, f"something wrong!"

        # Step III: 获得变量名
        name: Optional[str] = None
        signpost3 = signpost2
        for token in reversed(
                list(filter(lambda x: x[2] == Token.Name, self.__tokens_unprocessed[start_iloc: signpost2]))):
            name = token[3]
            signpost3 = token[0]
            if not (regex.match(r"^__", name) or regex.match(r"^_\w+_$", name)):
                break
        # special case: [OutAttribute]Int32 *outBegIdx
        if name is None:
            _u = -1
            _b = -1
            signpost2 = signpost1
            for token in reversed(
                    list(filter(lambda x: x[2] == Token.Name, self.__tokens_unprocessed[start_iloc: signpost2]))):
                name = token[3]
                signpost3 = token[0]
                if name[:2] != "__":
                    break
        # special case:
        is_reference = False
        _type = ""
        for t in self.__tokens_unprocessed[start_iloc: signpost3]:
            if t[3] == '&':
                is_reference = True
                _type += '*'
            else:
                _type += t[3]

        # Step IV: 构造字典
        if _u != -1 and _b != -1:
            sdecl = _type + '*' + \
                    self.__code[self.__tokens_unprocessed[signpost3][1]: self.__tokens_unprocessed[signpost2][1]] + \
                    self.__code[self.__tokens_unprocessed[_u][1]: self.__tokens_unprocessed[signpost1][1]]
            l1 = len(
                self.__code[self.__tokens_unprocessed[start_iloc][1]: self.__tokens_unprocessed[signpost3][1]] + '*')
            l2 = l1 + len(self.__tokens_unprocessed[signpost3][3])
        else:
            sdecl = _type + self.__code[
                            self.__tokens_unprocessed[signpost3][1]: self.__tokens_unprocessed[signpost1][1]]
            l1 = len(self.__code[self.__tokens_unprocessed[start_iloc][1]: self.__tokens_unprocessed[signpost3][1]])
            l2 = l1 + len(self.__tokens_unprocessed[signpost3][3])
        '''

        PCRE = r"\s*(?P<total>(?P<p3>(?12)((?:\(\*\s*(?P<paramName3_1>\w+)\)|(?P<paramName3_2>\w+))\s*(\((?P<param>(?>[^()\"\']+|(\"[^\"]*?\")|(\'[^\']+?\')|(?6))*)\)))\s*)|(?P<p1>((?P<paramType>((?P<qualifier>(const|volatile)\s+){0,2}(?P<special>(?:unsigned|signed|struct|union|enum)\s+)?(?P<namespace>\w+::)?\w+\s*?((\(((?>[^()]+|(?19))*)\))\s*)?((<((?>[^<>]+|(?22))*)>)\s*)?)(\*\s*(const\s*)?)*(\*\s*(const\s+)?|(?P<quote>&)\s*|\s+)?)(?P<paramName>\w+)\s*(?P<array>\[[^\]]*\]\s*)?))|(?P<p5>(?12)(\((?>\**(?P<fptr>\w+)|(?32))*\)\s*(\(((?>[^()]+|(?34))*)\)))\s*)|(?P<p2>\.\.\.)\s*|(?P<p4>void)\s*)(?P<attr>\w+\s*)?(=\s*(?P<intialValue>[^,]+)\s*)?(?:$|,)"

        res = {}
        for r in regex.finditer(PCRE, code):
            sdecl = r.group("total")
            if sdecl.strip() == "void":
                continue
            if r.group("paramName") is not None:
                name = r.group("paramName")
                loc = r.span("paramName")
            elif r.group("paramName3_1") is not None:
                name = r.group("paramName3_1")
                loc = r.span("paramName3_1")
            elif r.group("paramName3_2") is not None:
                name = r.group("paramName3_2")
                loc = r.span("paramName3_2")
            elif r.group("fptr") is not None:
                name = r.group("fptr")
                loc = r.span("fptr")
            else:
                raise
            if r.group("array") is not None:
                _loc = r.span("array")
                sdecl = sdecl[:loc[0]]+'*'+sdecl[loc[1]:_loc[0]]+sdecl[_loc[1]:]
                loc = (loc[0]+1, loc[1]+1)
            default_value = r.group("intialValue")
            is_reference = r.group("quote") is not None
            if is_reference:
                _loc = r.span("quote")
                sdecl = sdecl[:_loc[0]] + '*' + sdecl[_loc[1]:]
            res[name] = ParamDecl(sdecl, loc, default_value, is_reference)
        return res

    def __split_params(self, start_iloc: int, end_iloc: int) -> str:
        """
        采用栈思想，在匹配符号内的 `,` 不计入分割。
        :param start_iloc: 参数开始位置，不包括 ()
        :param end_iloc: 参数结束位置（开闭规则：左闭右开）
        :return: last_iloc, iloc
        """
        # print("total: ", self.__code[self.__tokens_unprocessed[start_iloc][1]: self.__tokens_unprocessed[end_iloc][1]])

        count = 0
        split_left = ['(', '{', '<', '[']
        split_right = [')', '}', '>', ']']
        last_iloc = start_iloc
        # 在 param 定义中不会出现比较运算等，所以不用判断类型了，不然会显示 operator
        # print("tks: ", self.__tokens_unprocessed[start_iloc: end_iloc])
        filtered_range = list(filter(lambda iloc: not self.__tokens_unprocessed[iloc][2] == Token.Text.Whitespace,
                                     range(start_iloc, end_iloc)))
        if len(filtered_range) > 0:
            end_iloc = max(filtered_range)
        for iloc in filtered_range:
            if self.__tokens_unprocessed[iloc][2] == Token.Punctuation and self.__tokens_unprocessed[iloc][3] in split_left:
                count += 1
            elif self.__tokens_unprocessed[iloc][2] == Token.Punctuation and self.__tokens_unprocessed[iloc][3] in split_right:
                count -= 1
            if count == 0:
                if self.__tokens_unprocessed[iloc][2] == Token.Punctuation and self.__tokens_unprocessed[iloc][3] == ',':
                    yield last_iloc, iloc
                    last_iloc = iloc + 1
                elif iloc == end_iloc:
                    yield last_iloc, iloc + 1

    def __get_function_body(self, iloc: int):
        """
        获得函数体
        核心思想：与语法分析器相同的栈思想 {} 匹配
        :param iloc: 函数名在 self.__tokens_unprocessed 中的索引
        :return:
        """
        block_boundaries = filter(lambda t: t[0] >= iloc and t[2] == Token.Punctuation and
                                            (t[3] == '{' or t[3] == '}' or t[3] == ';'),
                                  self.__tokens_unprocessed)
        count = 0
        start_iloc = -1
        end_iloc = -1
        for round_bracket in block_boundaries:
            if round_bracket[3] == '{':
                if count == 0:
                    start_iloc = round_bracket[0]
                count += 1
            elif round_bracket[3] == '}':
                end_iloc = round_bracket[0]
                count -= 1
            elif count == 0:
                # 识别函数有一些bug，例如`unsigned int (*spi_write)(void *, const char *, int);`中的`int`会被识别为function
                # 这一行的意思是如果在没有进入 {} 前遇到了 ; ，则说明匹配不成功
                # 也包括函数声明但不定义的额情况
                break
            if count == 0:
                break
        if start_iloc == -1 or end_iloc == -1 or count != 0:
            raise InvalidFormat()
        # end_token = self.__tokens_unprocessed[end_iloc]
        return start_iloc, end_iloc + 1

    def list_all_functions(self):
        """
        列出所有函数名
        :return: List[(iloc: int, sloc: int, name: str),...]
        """
        return list(map(lambda r: (r[0], r[1], r[3]),
                        filter(lambda t: t[2] == Token.Name.Function, self.__tokens_unprocessed)))

    def is_overlap_in_function_name(self):
        func_names = list(map(lambda x: x[2], self.list_all_functions()))
        return len(func_names) != len(set(func_names))

    def get_function(self, iloc: int, rois: List[int] = None, _params: Dict = None, cache: bool = False) \
            -> Tuple[bool, Optional[bool], Optional[FunctionDeclOrig], Optional[FunctionDecl]]:
        funcName = self.__tokens_unprocessed[iloc][3]
        # print(f"====================funcName={funcName}")
        if funcName in self.__cache:
            return True, self.__cache[funcName][0], self.__cache[funcName][1], self.__cache[funcName][2]
        try:
            is_enhanced = True  # 不能调动位置，后面如果遇到 ... 会需要设置为 False

            istart, iend = self.__get_function_return_type(iloc)
            _s = istart
            start = self.__tokens_unprocessed[istart][1]
            end = self.__tokens_unprocessed[iend][1]
            returnType = self.__code[start:end].strip()
            istart, iend = self.__get_function_param_decl(iloc)
            # print("pdecl=", self.__tokens_unprocessed[istart+1:iend-1])
            code = ''.join(map(lambda x: x[3], self.__tokens_unprocessed[istart+1: iend-1]))
            params = {}
            if "..." in code:
                is_enhanced = False
            else:
                params = self.__parse_param_decl(code)
            # print(f"k={k}")
            #     print(f"v={v}")

            istart, iend = self.__get_function_body(iend)
            _s_body = istart
            _e = iend
            funcBody = list(map(lambda t: [t[2], t[3]], self.__tokens_unprocessed[istart: iend]))
            try:
                funcBody = structural_normalize(funcBody)
            except EnhancementFailure as e:
                # print('+'*50)
                # print('istart', istart)
                # print('iend', iend)
                # print(self.__tokens_unprocessed)
                is_enhanced = False
            # 函数体的行号范围，不能统计 __tokens_unprocessed 中的行号，因为存在 multiline comment
            start_line = ''.join(map(lambda x: x[3], self.__tokens_unprocessed[:_s_body])).count('\n')
            end_line = start_line + ''.join(map(lambda x: x[3], self.__tokens_unprocessed[_s_body:_e])).count('\n')
            if rois is not None:
                cur_roi = list(map(lambda x: x - start_line, filter(lambda x: start_line <= x <= end_line, rois)))
            else:
                cur_roi = None
            line_number_in_file = ''.join(map(lambda x: x[3], self.__tokens_unprocessed[:_s])).count('\n')
            line_number_before_body = ''.join(map(lambda x: x[3], self.__tokens_unprocessed[_s: _s_body])).count('\n')
            # cur_roi 是从 function 的 body 开始计算的行号
            f = FunctionDecl(returnType, funcName, params, funcBody[1:-1], (start_line, end_line),
                             line_number_before_body, cur_roi)
            fo = FunctionDeclOrig(list(map(lambda x: [x[2], x[3]], self.__tokens_unprocessed[_s: _s_body])) + funcBody,
                                  funcName, _s_body - _s, (start_line, end_line), line_number_in_file,
                                  line_number_before_body, cur_roi)
        except InvalidFormat:
            return False, None, None, None
        if cache:
            self.__cache[funcName] = (is_enhanced, fo, f)
        # print(f"{fo.fname} is_enhanced = {is_enhanced}")
        return True, is_enhanced, fo, f

    def set_function_name_by_pcre(self):
        PCRE_FuncDecl = r"((static|inline|extern)\s*){0,3}(?P<attr>\w+\s*)?(?P<returnType>(const\s+)?(?P<special>(?:unsigned|struct|union|enum)\s+)?(?P<namespace>\w+::\s*)?\w+\s*(\*\s*(const\s*)?)*(\*\s*(const\s+)?|(<(?>[^<>]+|(?12))>\s*)&?\s*|(?P<quote>&)\s*|\s+))((?P<className>\w+(<(?>[^<>]+|(?16))>\s*)?::\s*)*(?P<funcName>(~\s*)?\w+[^(\s]?))\s*((\((?P<param>(?>[^()\"\']+|(\"[^\"]*?\")|(\'[^\']+?\')|(?20))*)\))\s*)()(?P<afterconst>const\s*)?(?3)?({(?P<funcBody>(?>[^{}]+|(?26))*)})"

        for t in filter(lambda item: item[2] == Token.Name.Function, self.__tokens_unprocessed):
            self.__tokens_unprocessed[t[0]] = (t[0], t[1], Token.Name, t[3])

        for res in regex.finditer(PCRE_FuncDecl, self.__code):
            # [iloc, sloc, type, string]
            sloc = res.start("funcName")
            for t in filter(lambda item: item[1] == sloc, self.__tokens_unprocessed):
                self.__tokens_unprocessed[t[0]] = (t[0], t[1], Token.Name.Function, t[3])


def test():
    code = r"""
    int foo1(int &a) {
        int b = a++;
        if (b>0) return b;//
        else return ++a;
    }

    int foo2(int &a) {
        int b = a++;
        if (b>0) return b;
        else return ++a;//
    }

    int main() {
        int a = 0;
        cout << foo1(a) << foo2(a) << endl;//
        foo3();
        cout << a << endl;
        return 0;
    }

    void foo3() {
    // fsdafsdf
    cout << "foo3" << endl;
    return;
    }

    void rhizome_ectomeres(int clamming_minorca,union georgetta_cumin unwreathed_berrie)
    {
        FILE *stonesoup_fpipe;
        char stonesoup_buffer[100];
        char stonesoup_command_buffer[1000];
        char *stonesoup_command_str = "nslookup ";
      char *bert_tiburon = 0;
      ++stonesoup_global_variable;
      clamming_minorca--;
      if (clamming_minorca > 0) {
        rhizome_ectomeres(clamming_minorca,unwreathed_berrie);
        return ;
      }
      bert_tiburon = ((char *)unwreathed_berrie . celiolymph_orgamy);
        tracepoint(stonesoup_trace, weakness_start, "CWE078", "A", "Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')");
        if (strlen(bert_tiburon) < 1000 - strlen(stonesoup_command_str)) {
            tracepoint(stonesoup_trace, variable_buffer, "STONESOUP_TAINT_SOURCE", bert_tiburon, "INITIAL-STATE");
            tracepoint(stonesoup_trace, trace_point, "CROSSOVER-POINT: BEFORE");
            /* STONESOUP: CROSSOVER-POINT (OS Command Injection) */
            snprintf(stonesoup_command_buffer, 1000, "%s%s",stonesoup_command_str,bert_tiburon);
            tracepoint(stonesoup_trace, variable_buffer, "stonesoup_command_buffer", stonesoup_command_buffer, "CROSSOVER-STATE");
            tracepoint(stonesoup_trace, trace_point, "CROSSOVER-POINT: AFTER");
            tracepoint(stonesoup_trace, trace_point, "TRIGGER-POINT: BEFORE");
            /* STONESOUP: TRIGGER-POINT (OS Command Injection) */
            stonesoup_fpipe = popen(stonesoup_command_buffer,"r");
            if (stonesoup_fpipe != 0) {
                while(fgets(stonesoup_buffer,100,stonesoup_fpipe) != 0) {
                    stonesoup_printf(stonesoup_buffer);
                    return ;
                }
                pclose(stonesoup_fpipe);
            }
            tracepoint(stonesoup_trace, trace_point, "TRIGGER-POINT: AFTER");
        }
        tracepoint(stonesoup_trace, weakness_end);
    ;
      if (unwreathed_berrie . celiolymph_orgamy != 0) 
        free(((char *)unwreathed_berrie . celiolymph_orgamy));
    stonesoup_close_printf_context();
    }
    """
    fe = FExtracter(code, 'C++')
    caller_candidates = ["main"]
    callee_candidates = ["foo1", "foo2", "foo3"]
    # caller_candidates = ["rhizome_ectomeres"]
    # callee_candidates = ["rhizome_ectomeres"]
    callers = []
    callees = []
    for f in fe.list_all_functions():
        if f[2] in caller_candidates + callee_candidates:
            _, flag, fp0, fp1 = fe.get_function(f[0], rois=[3, 10, 15])
            if f[2] in caller_candidates:
                callers.append(fp0)
            if f[2] in callee_candidates:
                callees.append(fp1)
    # for caller in f_caller:
    #     for callee in f_callee:
    #         print("=" * 10)
    #         flag, result, roi = caller.expand(callee)
    #         print(f"===== caller: {caller.fname} callee: {callee.fname} flag: {flag} =====")
    #         if flag:
    #             print(result)
    #             print(roi)
    #             # print("caller.tokens=", caller.tokens)
    for caller in callers:
        print("=" * 10)
        for callee in callees:
            flag, result, roi = caller.expand(callee)
            if flag:
                print(result)
                print(roi)


if __name__ == "__main__":
    # import glob
    # for filename in glob.glob(r"/Users/huanghongjun/Desktop/FiCoVuL/preprocess/data/raw/FUNDED_GitHub/**/**/*.txt"):
    #     print(filename)
    #     with open(filename, 'r') as fp:
    #         code = fp.read()
    #     fe = FExtracter(code, 'c')
    #     for f in fe.list_all_functions():
    #         try:
    #             fp = fe.get_function(f[0])
    #             print(fp)
    #         except InvalidFormat:
    #             pass

    # with open(
    #         "/Users/huanghongjun/Desktop/FiCoVuL/preprocess/data/raw/FUNDED_GitHub/CWE-369/old_files/ee972197b670dea32b7826835c3966725fcec777.txt",
    #         'r') as fp:
    #     code = fp.read()
    # fe = FExtracter(code, 'c')
    # for f in fe.list_all_functions():
    #     try:
    #         fp = fe.get_function(f[0])
    #         print(fp)
    #     except InvalidFormat:
    #         print("InvalidFormat")

    test()

    # with open(
    #         r"/Users/huanghongjun/Desktop/FiCoVuL/preprocess/data/raw/FUNDED_GitHub/CWE-665/old_files/14558a9fa3b3a5510a1d360b38935bf2c8296d9b.txt",
    #         'r') as fp:
    #     code = fp.read()
    # fe = FExtracter(code, 'c')
    # for f in fe.list_all_functions():
    #     try:
    #         print(f"===== {f[2]} =====")
    #         fp = fe.get_function(f[0], list(map(lambda x: x-1, [])))
    #         print(fp)
    #     except InvalidFormat:
    #         print("InvalidFormat")
