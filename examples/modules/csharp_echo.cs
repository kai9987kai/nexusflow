using System;
using System.Text;

class Program
{
    static string Escape(string s)
    {
        if (s == null) return "";
        return s.Replace("\\", "\\\\").Replace("\"", "\\\"");
    }

    static void Main(string[] args)
    {
        var sb = new StringBuilder();
        sb.Append("{\"lang\":\"csharp\",\"args\":[");
        for (int i = 0; i < args.Length; i++)
        {
            if (i > 0) sb.Append(",");
            sb.Append("\"").Append(Escape(args[i])).Append("\"");
        }
        sb.Append("],\"count\":").Append(args.Length).Append("}");
        Console.WriteLine(sb.ToString());
    }
}

