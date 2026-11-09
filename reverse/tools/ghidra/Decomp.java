import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.*;
import ghidra.program.model.address.*;
import ghidra.program.model.listing.*;
import ghidra.app.cmd.function.CreateFunctionCmd;
import java.io.*;
public class Decomp extends GhidraScript {
  public void run() throws Exception {
    String[] a = getScriptArgs();
    String out = a[0];
    PrintWriter pw = new PrintWriter(new FileWriter(out, true));
    DecompInterface di = new DecompInterface();
    DecompileOptions opt = new DecompileOptions();
    di.setOptions(opt);
    di.openProgram(currentProgram);
    for (int i=1;i<a.length;i++){
      Address ad = toAddr(a[i]);
      Function f = getFunctionAt(ad);
      if (f==null){
        CreateFunctionCmd c = new CreateFunctionCmd(ad);
        c.applyTo(currentProgram);
        f = getFunctionAt(ad);
      }
      if (f==null){ pw.println("// no function at "+a[i]); continue;}
      DecompileResults r = di.decompileFunction(f, 120, monitor);
      pw.println("// ===== "+a[i]+" "+f.getName());
      if (r.decompileCompleted()) pw.println(r.getDecompiledFunction().getC());
      else pw.println("// fail "+r.getErrorMessage());
    }
    pw.close();
  }
}
