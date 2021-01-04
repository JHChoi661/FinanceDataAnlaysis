from slacker import Slacker

slack = Slacker('xoxb-1600521518790-1613463840660-SEossYKo1khLpOjXMScT0sFX')


attach_dict = {
    'color' : '#ff0000',
    'author_name' :'INVESTAR',
    'author_link' :'https://github.com/JHChoi661/myFirstRepository',
    'title' :'오늘의 증시 KOSPI',
    'title_link' :'https://finance.naver.com/sise/sise_index.nhn?code=KOSPI',
    'text' :'2,873.47 ^52.96 (+1.88%)',
    'image_url' :'https://ssl.pstatic.net/imgstock/chart3/day/KOSPI.png?sidcode=1609486003467'
}

attach_list = [attach_dict]
slack.chat.post_message(channel='#investing',attachments=attach_list)